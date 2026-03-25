"""Utilities to evaluate tool-calling accuracy for adapter-based LLMs.

This module parses model outputs into structured tool calls, runs generation
on evaluation prompts, and reports aggregated accuracy metrics.
"""

import re
import sys
import json
from pathlib import Path
from typing import List, Dict
import torch

# ==================== TOOL CALL PARSING ====================

def _qwen3_tool_call_parser(text: str) -> List[Dict]:
    """Extract tool calls from Qwen3 output markup.

    The parser expects one or more blocks in this format:
    ``<tool_call>{"name": "func", "arguments": {...}}</tool_call>``.
    Invalid JSON blocks are ignored.

    Args:
        text: Raw model output text.

    Returns:
        A list of parsed tool call dictionaries.
    """
    tool_calls = []
    pattern = r'<tool_call>(.*?)</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            tool_call = json.loads(match)
            tool_calls.append(tool_call)
        except json.JSONDecodeError:
            continue
    return tool_calls


def _functiongemma_tool_call_parser(raw_output: str) -> list[dict]:
    """Extract tool calls from FunctionGemma output markup.

    Expected format per call:
    ``<start_function_call>call:func_name{arg:<escape>value<escape>}<end_function_call>``.

    Args:
        raw_output: Raw model output text.

    Returns:
        A list of dictionaries with ``name`` and ``arguments`` keys.
    """
    pattern = r"<start_function_call>call:(\w+)\{(.*?)\}<end_function_call>"
    matches = re.findall(pattern, raw_output, re.DOTALL)
    tool_calls: list[dict] = []
    for func_name, args_str in matches:
        arguments = _parse_functiongemma_arguments(args_str)
        tool_calls.append({"name": func_name, "arguments": arguments})
    return tool_calls


def _parse_functiongemma_arguments(args_str: str) -> dict:
    """Parse FunctionGemma argument payloads into a dictionary.

    Arguments are expected as ``key:<escape>value<escape>`` pairs. Each value
    is first parsed as JSON; if JSON parsing fails, the raw string is kept.

    Args:
        args_str: Raw argument segment captured from model output.

    Returns:
        A dictionary of parsed argument names and values.
    """
    arguments = {}
    if not args_str.strip():
        return arguments
    args_pattern = r"(\w+):<escape>(.*?)<escape>"
    matches = re.findall(args_pattern, args_str, re.DOTALL)
    for key, value in matches:
        try:
            parsed_value = json.loads(value)
            arguments[key] = parsed_value
        except json.JSONDecodeError:
            arguments[key] = value
    return arguments

# ==================== UTILS ====================

def _resolve_base_model(adapter_path: str, override: str | None) -> str:
    """Resolve the base model identifier associated with an adapter.

    Resolution order:
    1. ``override`` argument, when provided.
    2. ``base_model_name_or_path`` from ``adapter_config.json``.

    Args:
        adapter_path: Directory containing adapter files.
        override: Optional explicit base model identifier.

    Returns:
        The resolved base model identifier.

    Exits:
        Terminates the process with an error message if the adapter config is
        missing or does not include ``base_model_name_or_path``.
    """
    if override:
        return override
    config_path = Path(adapter_path) / "adapter_config.json"
    if not config_path.exists():
        sys.exit(
            f"[ERROR] adapter_config.json not found in {adapter_path}.\n"
            "Provide --base-model explicitly."
        )
    with open(config_path) as f:
        cfg = json.load(f)
    base = cfg.get("base_model_name_or_path")
    if not base:
        sys.exit("[ERROR] base_model_name_or_path not in adapter_config.json.\n"
                 "Provide --base-model explicitly.")
    print(f"[info] Base model detected from adapter_config.json: {base}")
    return base

# ==================== EVALUATION ====================

def evaluate_tool_calling_accuracy(
    model,
    eval_dataset,
    processor,
    tools,
    model_type: str = "qwen3",
    batch_size: int = 8,
) -> Dict:
    """Evaluate tool-calling quality on a chat-formatted evaluation dataset.
 
    This function loads the base model and adapter from ``adapter_path``, builds
    prompt contexts for assistant tool-call turns, generates model outputs
    sample-by-sample (required for Unsloth's fast inference kernels), and
    compares predicted tool calls against expected calls.
 
    NOTE: ``batch_size`` is accepted for API compatibility but generation is
    always performed one sample at a time to avoid Unsloth's RoPE broadcast
    error with batched prefill (shape [B, H, seq_len, D] vs [B, H, 1, D]).
 
    Reported metrics:
    - ``tool_name_acc``: fraction of expected calls with correct tool name.
    - ``arg_exact``: fraction of expected calls whose arguments match exactly
      under recursive validation.
    - ``valid_json``: fraction of matched calls whose generated arguments are
      JSON-serializable.
 
    Args:
        adapter_path: Path to the adapter directory.
        eval_dataset: Evaluation conversations in chat-message format.
        processor: Chat processor/tokenizer used to format prompts (will be
            reloaded from ``adapter_path`` internally).
        tools: Tool schema list passed into the chat template.
        model_type: Output parser selector (``qwen3`` or ``functiongemma``).
        override_model_base: Optional base model identifier override.
        batch_size: Kept for API compatibility; generation runs sample-by-sample.
 
    Returns:
        A metrics dictionary with accuracy rates and total evaluated calls.
 
    Raises:
        ValueError: If ``model_type`` is not supported.
    """
 
    if model_type == "qwen3":
        extract_tool_calls_from_text = _qwen3_tool_call_parser
    elif model_type == "functiongemma":
        extract_tool_calls_from_text = _functiongemma_tool_call_parser
    else:
        raise ValueError(
            f"Unsupported model type: {model_type}, "
            f"valid types are ['qwen3', 'functiongemma']"
        )
 
    model.eval()
 
    # ------------------------------------------------------------------
    # 1. Collect ALL (context_text, expected_tool_calls) pairs up front
    # ------------------------------------------------------------------
    samples: list[tuple[str, list]] = []  # (prompt_text, expected_tool_calls)
 
    for messages in eval_dataset:
        tool_call_turns = [
            i
            for i, msg in enumerate(messages)
            if msg["role"] == "assistant"
            and "tool_calls" in msg
            and msg["tool_calls"]
        ]
 
        for turn_idx in tool_call_turns:
            context_messages = messages[:turn_idx]
            expected_tool_calls = messages[turn_idx].get("tool_calls", [])
 
            if not expected_tool_calls:
                continue
 
            context_text = processor.apply_chat_template(
                context_messages,
                tokenize=False,
                tools=tools,
                add_generation_prompt=True,
                continue_final_message=False,
                enable_thinking=False,
            )
 
            samples.append((context_text, expected_tool_calls))
 
    # ------------------------------------------------------------------
    # 2. Run batched generation
    # ------------------------------------------------------------------
    # processor / tokenizer must pad on the LEFT for decoder-only models
    # so that the last real token of every sample is right-aligned and
    # the model generates immediately after it.
    original_padding_side = processor.padding_side
    processor.padding_side = "left"
 
    all_generated_texts: list[str] = []
 
    print("\n" + "=" * 60)
    print("EVALUATING TOOL CALLING ACCURACY (batched)")
    print("=" * 60)
 
    prompts = [s[0] for s in samples]
 
    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start : batch_start + batch_size]
 
        # Tokenise the whole batch at once (padding handled automatically)
        inputs = processor(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)
 
        input_lengths = inputs["input_ids"].shape[1]  # same for all after padding
 
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=processor.pad_token_id,
                max_token_length=None,
                use_cache=False,
            )
 
        # Decode only the newly generated tokens for each item in the batch
        for i, output_ids in enumerate(outputs):
            generated_ids = output_ids[input_lengths:]
            generated_text = processor.decode(generated_ids, skip_special_tokens=True)
            all_generated_texts.append(generated_text)
            # print(f"Generated text for sample {batch_start + i + 1}:\n{generated_text}\n{'-'*40}")
 
        batch_end = min(batch_start + batch_size, len(prompts))
        print(f"Generated: {batch_end}/{len(prompts)}")
 
    # Restore original padding side
    processor.padding_side = original_padding_side
 
    # ------------------------------------------------------------------
    # 3. Score predictions
    # ------------------------------------------------------------------
    correct_tool_names = 0
    exact_arg_matches = 0
    valid_json_count = 0
    total_tool_calls = 0
 
    for (_, expected_tool_calls), generated_text in zip(samples, all_generated_texts):
        predicted_tool_calls = extract_tool_calls_from_text(generated_text)
 
        for expected_tc in expected_tool_calls:
            total_tool_calls += 1
 
            expected_name = expected_tc.get("name", "")
            expected_args = expected_tc.get("arguments", {})
 
            for pred_tc in predicted_tool_calls:
                pred_name = pred_tc.get("name", "")
                pred_args = pred_tc.get("arguments", {})
 
                if pred_name == expected_name:
                    correct_tool_names += 1
 
                    if _validate_tool_args(pred_args, expected_args):
                        exact_arg_matches += 1
                    else:
                        print(
                            f"Expected arguments:\n{expected_args}\nGenerated:\n{pred_args}"
                        )
 
                    try:
                        json.dumps(pred_args)
                        valid_json_count += 1
                    except Exception:
                        pass
 
                    break
 
    # ------------------------------------------------------------------
    # 4. Report
    # ------------------------------------------------------------------
    results = {
        "tool_name_acc": correct_tool_names / total_tool_calls if total_tool_calls > 0 else 0,
        "arg_exact": exact_arg_matches / total_tool_calls if total_tool_calls > 0 else 0,
        "valid_json": valid_json_count / total_tool_calls if total_tool_calls > 0 else 0,
        "total_evaluated": total_tool_calls,
    }
 
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS:")
    print("=" * 60)
    print(f"Tool Name Accuracy:    {results['tool_name_acc']:.2%}")
    print(f"Argument Exact Match:  {results['arg_exact']:.2%}")
    print(f"Valid JSON Rate:       {results['valid_json']:.2%}")
    print(f"Total tool calls evaluated: {results['total_evaluated']}")
    print("=" * 60 + "\n")
 
    return results


def _validate_tool_args(generated_args: Dict, expected_args: Dict) -> bool:
    """Recursively validate generated tool arguments against expectations.

    The validation is key-oriented from generated arguments to expected
    arguments. It checks that every generated key exists in expected data and
    that values match exactly, including nested dictionaries.

    Args:
        generated_args: Arguments predicted by the model.
        expected_args: Ground-truth arguments from the dataset.

    Returns:
        ``True`` when all generated keys and values match; otherwise ``False``.
    """
    for key, gen_value in generated_args.items():
        exp_value = expected_args.get(key, None)

        if exp_value is None:
            print(f"Unexpected argument: {key}")
            return False

        if isinstance(gen_value, dict) and isinstance(exp_value, dict):
            if not _validate_tool_args(gen_value, exp_value):
                print(f"Mismatch in nested argument: {key}")
                return False
        else:
            if gen_value != exp_value:
                print(f"Argument value mismatch for '{key}': expected '{exp_value}', got '{gen_value}'")
                return False

    return True
