import torch
import wandb
import os
from dataclasses import dataclass
from typing import Any, Dict, List
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoProcessor
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import json
from typing import Dict, List
from eval import evaluate_tool_calling_accuracy

model_id = "Qwen/Qwen3-0.6B"
os.environ["HF_TOKEN"] = "hf_xxxxxxx"  # Set your Hugging Face token here or ensure it's set in your environment variables

# ==================== DATASET PREPARATION ====================

@dataclass
class DataCollatorForChatML:
    """
    Data collator for chat-based models that applies dynamic padding and creates 
    labels for the assistant's responses. Assumes that the processor's apply_chat_template 
    method has been used to create input_ids and assistant_masks.
    """
    processor: AutoProcessor
    padding: bool = True
    max_length: int = 8192
    pad_to_multiple_of: int = 8  # for better GPU memory alignment
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Find the maximum sequence length in the batch
        max_len = max(len(f["input_ids"]) for f in features)
        
        # Optionally round up to the nearest multiple
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) 
                      // self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        # Limit to max_length
        max_len = min(max_len, self.max_length)
        
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }
        
        pad_token_id = self.processor.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processor.eos_token_id
        
        for feature in features:
            # Get the sequences
            input_ids = feature["input_ids"][:max_len]
            attention_mask = feature["attention_mask"][:max_len]
            labels = feature["labels"][:max_len]
            
            # Calculate how much padding we need
            padding_length = max_len - len(input_ids)
            
            # Apply right padding
            input_ids = input_ids + [pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            labels = labels + [-100] * padding_length  # -100 is ignored in loss
            
            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(labels)
        
        # Convert to tensors
        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long)
        }

def load_custom_system_prompt(txt_path: str) -> str:
    """Loads a custom system prompt from a .txt file"""
    with open(txt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def replace_system_prompt(messages: List[Dict], new_system_prompt: str) -> List[Dict]:
    """Replaces the system prompt in the conversation with the new one"""
    modified_messages = []
    for msg in messages:
        if msg["role"] == "system":
            modified_messages.append({
                "role": "system",
                "content": new_system_prompt
            })
        else:
            modified_messages.append(msg)
    return modified_messages

def prepare_dataset(
    json_path: str,
    system_prompt_path: str,
    processor: AutoProcessor,
    train_split: float = 0.8
):
    """
    Prepares the dataset for training
    
    Args:
        json_path: Path to the JSON with the data
        system_prompt_path: Path to the .txt with the system prompt
        processor: Processor of the model
        train_split: Proportion for train (0.8 = 80% train, 20% eval)
    
    Returns:
        train_dataset, eval_dataset
    """
    # Load custom system prompt
    custom_system_prompt = load_custom_system_prompt(system_prompt_path)
    print(f"System prompt loaded from {system_prompt_path}")
    print(f"Preview: {custom_system_prompt[:100]}...")
    
    # Load data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        data = data["train"]  # assuming the JSON has a top-level "train" key
    
    print(f"\nData loaded: {len(data)} examples")
    
    # Process and tokenize each example manually
    tokenized_data = []
    
    for idx, example in enumerate(data):
        # Replace system prompt and store the messages structure
        messages = replace_system_prompt(example["messages"], custom_system_prompt)
        # Apply chat template manualmente para este ejemplo
        tokenized = processor.apply_chat_template(
            messages,  
            return_assistant_tokens_mask=True,
            return_dict=True,
            padding=False,  # No padding yet, we'll do it in batches
            truncation=True,
            max_length=8192,  # Adjust as needed
            add_generation_prompt=False,
        )
        if not "labels" in tokenized:
            if tokenized["assistant_masks"] is not None:
                labels = [
                    input_id if mask else -100 
                    for input_id, mask in zip(tokenized["input_ids"], tokenized["assistant_masks"])
                ]
                # Extract tokenized data (they are in lists, take the first element)
                tokenized_data.append({
                    "input_ids": tokenized["input_ids"],
                    "attention_mask": tokenized["attention_mask"],
                    "labels": labels,
                    "assistant_masks": tokenized["assistant_masks"]
                })
            else:
                print(Warning("Could not compute labels from input_ids and assistant_masks because one of them is missing. Check the processor output."))
                continue
        
        # Print progress
        if idx == 0 or (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(data)} examples")
    
    print(f"\nTokenization complete!")
    
    # Verify first example
    print("\n" + "="*60)
    print("TOKENIZED EXAMPLE:")
    print("="*60)
    print(f"Input IDs example: {tokenized_data[0]['input_ids']}")
    print(f"Labels example: {tokenized_data[0]['labels']}")
    print(f"Decoded text preview:\n{processor.decode(tokenized_data[0]['input_ids'])}")
    print("="*60 + "\n")
    
    # Split train/eval
    split_idx = int(len(tokenized_data) * train_split)
    train_data = tokenized_data[:split_idx]  
    eval_data = tokenized_data[split_idx:]
    raw_eval_data = data[split_idx:]  # Keep the raw eval data for custom evaluation later
    
    print(f"Split: {len(train_data)} train, {len(eval_data)} eval")
    
    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)
    
    return train_dataset, eval_dataset, raw_eval_data

# ==================== TRAINING FUNCTION ====================

def train():
    """Main training function with wandb sweep integration"""
    # Initialize wandb sweep
    wandb.init(
        entity="oscarponsfdez-university-of-malaga",
        project="sft_bt_tools",
    )
    config = wandb.config  # get sweep hyperparameters

    # Create a unique run name based on hyperparameters for better tracking
    run_name = (
        f"r{config.lora_r}"
        f"_a{config.lora_alpha}"
        f"_lr{config.learning_rate:.0e}"
        f"_bs{config.gradient_accumulation_steps}"
    )
    wandb.run.name = run_name

    print("\n" + "="*60)
    print("SWEEP CONFIGURATION:")
    print("="*60)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("="*60 + "\n")

    # Load tokenizer
    processor = AutoProcessor.from_pretrained(model_id)

    # Load custom chat template from .jinja file
    with open("../templates/qwen3.jinja", "r", encoding="utf-8") as f:
        custom_template = f.read()
    processor.chat_template = custom_template
    
    # Prepare dataset
    train_dataset, eval_dataset, raw_eval_data = prepare_dataset(
        json_path="../data/train_dataset.json",  
        system_prompt_path="../data/system_prompt.txt",  
        processor=processor,
        train_split=0.8
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # attn_implementation="flash_attention_2"
    )
    print(f"\nModel loaded: {model}")
    # LoRA config with sweep hyperparameters
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training arguments with sweep hyperparameters
    training_args = TrainingArguments(
        output_dir=f"../results/{wandb.run.name}",
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=1,  # Keep this fixed to lower value and use gradient accumulation so vram does not explode
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="wandb",
        bf16=True,
        gradient_checkpointing=False,
        optim="adamw_torch_fused",
        weight_decay=0.01,
        max_grad_norm=1.0,
        # dataset_text_field="messages",
        # assistant_only_loss=True,
        # packing=False,
        # remove_unused_columns=False,
    )
    
    # Create data collator instance
    data_collator = DataCollatorForChatML(
        processor=processor,
        padding=True,
        max_length=8192,
        pad_to_multiple_of=8
    )
    # Trainer
    print("\n Setting up trainer...")
    # print(train_dataset)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=processor,
    )
    
    # Train
    print("\n Training is about to start!...\n")
    trainer.train()
    
    # Custom evaluation for tool calling accuracy
    print("\n Evaluating tool calling accuracy...")
    eval_results = evaluate_tool_calling_accuracy(
        trainer.model, 
        raw_eval_data,
        processor
    )
    
    # Log custom metrics
    wandb.log({
        "final/tool_name_accuracy": eval_results["tool_name_acc"],
        "final/arg_exact_match": eval_results["arg_exact"],
        "final/valid_json_rate": eval_results["valid_json"]
    })
    
    # Save the best model
    best_model_path = f"../models/{wandb.run.name}_best"
    trainer.save_model(best_model_path)
    print(f"\n Best model saved at: {best_model_path}")
    
    wandb.finish()

if __name__ == "__main__":
    train()