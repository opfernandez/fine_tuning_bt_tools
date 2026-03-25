from pathlib import Path
import torch
import wandb
import dotenv
import json
import os
import shutil
import unsloth
from transformers import Trainer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments,  AutoTokenizer
from unsloth import FastLanguageModel
from trl import SFTTrainer
from peft import LoraConfig
from eval import evaluate_tool_calling_accuracy
from data_loader import DataCollatorForChatML, prepare_dataset

dotenv.load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# model_id = "unsloth/functiongemma-270m-it"
model_id = "Qwen/Qwen3-0.6B"
model_type = "qwen3"  # for logging and naming purposes, e.g. "qwen3" or "functiongemma"

# Load tool descriptions from JSON file
try:
    with open(os.path.join("../data", "domotic_tools.json"), "r", encoding="utf-8") as f:
        bt_tool = json.load(f)
    print(f"Loaded {len(bt_tool)} tool(s) from domotic_tools.json")
except FileNotFoundError:
    raise FileNotFoundError("domotic_tools.json not found at ../data/domotic_tools.json. "
                            "Please create it before running training.")
except json.JSONDecodeError as e:
    raise ValueError(f"domotic_tools.json contains invalid JSON: {e}")

class EarlyStoppingCallback(TrainerCallback):
    """
    Early stopping callback stops training if eval loss does not improve for a
    specified number of evaluations (patience) by a certain threshold.
    It also marks the run as preempting in wandb to signal that the run was stopped early.
    """

    def __init__(self, patience: int = 2, threshold: float = 0.05):
        self.patience = patience
        self.threshold = threshold
        self.best_loss = float("inf")
        self.evals_without_improvement = 0
        self.stopped_early = False

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict,
        **kwargs,
    ) -> TrainerControl:

        current_loss = metrics.get("eval_loss")
        if current_loss is None:
            return control

        improvement_threshold = self.best_loss * (1 - self.threshold)
        if current_loss < improvement_threshold:
            self.best_loss = current_loss
            self.evals_without_improvement = 0
        else:
            self.evals_without_improvement += 1
            print(
                f"[EarlyStopping] No improvement {self.evals_without_improvement}/{self.patience} "
                f"(best: {self.best_loss:.4f}, current: {current_loss:.4f}, "
                f"threshold: {improvement_threshold:.4f})"
            )

        if self.evals_without_improvement >= self.patience:
            print(
                f"[EarlyStopping] Stopping training after {self.patience} "
                f"evaluations without improvement."
            )
            self.stopped_early = True
            if wandb.run is not None:
                wandb.mark_preempting()
            control.should_training_stop = True

        return control


# ==================== TRAINING FUNCTION ====================

def train():
    """Main training function with wandb sweep integration"""
    wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project=os.getenv("WANDB_PROJECT"),
        mode="online",
    )
    config = wandb.config
    run_name = (
        f"ep{config.num_train_epochs}"
        f"_r{config.lora_r}"
        f"_a{config.lora_alpha}"
        f"_lr{config.learning_rate:.0e}"
        f"_bs{config.gradient_accumulation_steps}"
    )
    wandb.run.name = run_name

    print("\n" + "=" * 60)
    print("SWEEP CONFIGURATION:")
    print("=" * 60)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 60 + "\n")

    # Load model + tokenizer via Unsloth 
    model, processor = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=1024,          # must match DataCollator / prepare_dataset
        dtype=torch.bfloat16,         # None lets Unsloth auto-detect
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
        token=os.getenv("HF_TOKEN"),
    )

    # Load custom chat template from .jinja file
    with open(os.path.join("../templates", f"{model_type}.jinja"), "r", encoding="utf-8") as f:
        custom_template = f.read()
    processor.chat_template = custom_template

    # Prepare dataset
    train_dataset, eval_dataset, raw_eval_data = prepare_dataset(
        json_path=os.path.join("../data", "domotic_dataset.json"),
        system_prompt_path=os.path.join("../data", "system_prompt_domotic.jinja"),
        processor=processor,
        tools=bt_tool,
        train_split=0.8,
        max_length=1024,
    )

    # Wrap model with LoRA via Unsloth
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        # use_gradient_checkpointing="unsloth",   # Unsloth's memory-efficient variant
        random_state=42,
        use_rslora=False,   # set True to use Rank-Stabilised LoRA
        loftq_config=None,
    )
    model.print_trainable_parameters()

    output_dir = os.path.join("../checkpoints", wandb.run.name)

    # compute gradient_accumulation_steps based on batch size and desired effective batch size
    if config.gradient_accumulation_steps <= config.max_batch_size:
        bs = config.gradient_accumulation_steps
        gradient_accumulation_steps = 1
    else:
        bs = config.max_batch_size
        gradient_accumulation_steps = config.gradient_accumulation_steps // bs  # this gives the number of steps to accumulate to reach the effective batch size
    print(f"Using per_device_train_batch_size={bs} with gradient_accumulation_steps={gradient_accumulation_steps} to achieve effective batch size of {config.gradient_accumulation_steps}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=3,
        save_strategy="steps",
        save_steps=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="wandb",
        bf16=True,
        fp16=False,
        # gradient_checkpointing is handled by Unsloth above
        gradient_checkpointing=False,
        optim="adamw_8bit",          
        weight_decay=0.01,
        max_grad_norm=1.0,
    )

    data_collator = DataCollatorForChatML(
        processor=processor,
        padding=True,
        max_length=1024,
        pad_to_multiple_of=8,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=processor,
        callbacks=[EarlyStoppingCallback(patience=2, threshold=0.05)],
    )

    print("\n Training is about to start!...\n")
    trainer.train()

    # Post-training evaluation
    early_stopping_callback = next(
        (cb for cb in trainer.callback_handler.callbacks
         if isinstance(cb, EarlyStoppingCallback)),
        None,
    )

    should_evaluate = True
    eval_loss_threshold = 0.5

    if early_stopping_callback:
        print(f"\n Training stopped early with best loss: {early_stopping_callback.best_loss:.4f}")
        if early_stopping_callback.best_loss > eval_loss_threshold:
            print(f" Loss {early_stopping_callback.best_loss:.4f} > threshold {eval_loss_threshold:.4f}, skipping final evaluation")
            should_evaluate = False
        else:
            print(f" Loss {early_stopping_callback.best_loss:.4f} <= threshold {eval_loss_threshold:.4f}, proceeding with final evaluation")

    # Only save and evaluate the model if it achieved a reasonable eval loss (to avoid wasting resources on very bad models)
    if should_evaluate:
        model_eval = FastLanguageModel.for_inference(trainer.model)
        # Save adapter
        best_model_path = os.path.join("../adapters", f"{model_type}_{wandb.run.name}_best")
        # Save only the LoRA adapter weights (small, portable)
        trainer.model.save_pretrained(best_model_path)
        # Load base template from Hugging Face Hub and save it to the same directory (for easy loading with from_pretrained)
        base_processor = AutoTokenizer.from_pretrained(model_id)
        processor.chat_template = base_processor.chat_template
        processor.save_pretrained(best_model_path)
        print(f"\n Best model saved at: {best_model_path}")
        # Save GGUF
        best_gguf_path = os.path.join("../adapters", f"{model_type}_{wandb.run.name}_best_gguf")
        if config.export_to_q8:
            try:
                trainer.model.save_pretrained_gguf(
                    best_gguf_path,
                    processor,
                    quantization_method="q8_0",  # options: q4_k_m, q8_0, f16, …
                )
                print(f" Best Q8 GGUF saved at: {best_gguf_path}")
            except Exception as e:
                print(f"Error exporting to Q8 GGUF: {e}")
        if config.export_to_q4:
            try:
                trainer.model.save_pretrained_gguf(
                    best_gguf_path,
                    processor,
                    quantization_method="q4_k_m",  # options: q4_k_m, q8_0, f16, …
                )
                print(f" Best Q4 GGUF saved at: {best_gguf_path}_q4")
            except Exception as e:
                print(f"Error exporting to Q4 GGUF: {e}")

        # Final evaluation of tool-calling accuracy on the raw eval data using the best model
        eval_results = evaluate_tool_calling_accuracy(
            model=model_eval,
            eval_dataset=raw_eval_data,
            processor=processor,
            tools=bt_tool,
            model_type=model_type,
            batch_size=32,
        )
        wandb.log({
            "final/tool_name_accuracy": eval_results["tool_name_acc"],
            "final/arg_exact_match": eval_results["arg_exact"],
            "final/valid_json_rate": eval_results["valid_json"],
        })

    print("Deleting checkpoints to save space ...")
    try:
        shutil.rmtree(output_dir)
        print("Removed successfully ...")
    except OSError as error:
        print(error)
        print("Folder can not be removed")

    wandb.finish()


if __name__ == "__main__":
    train()
