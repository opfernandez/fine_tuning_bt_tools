import torch
import wandb
from transformers import TrainingArguments, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset, Dataset
import argparse
import json
from typing import Dict, List
from .eval import evaluate_tool_calling_accuracy

model_id = "Qwen/Qwen3-8B"

# ==================== DATASET PREPARATION ====================
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
    tokenizer,
    train_split: float = 0.8
):
    """
    Prepares the dataset for training
    
    Args:
        json_path: Path to the JSON with the data
        system_prompt_path: Path to the .txt with the system prompt
        tokenizer: Tokenizer of the model
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
    
    print(f"\nData loaded: {len(data)} examples")
    
    # Process each example
    processed_data = []
    
    for idx, example in enumerate(data):
        # Replace system prompt
        messages = replace_system_prompt(example["messages"], custom_system_prompt)
        
        # Apply model's chat template
        # IMPORTANT: add_generation_prompt=False to include the assistant's response
        try:
            formatted_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            processed_data.append({
                "text": formatted_text,
                "original_messages": messages  # saved for evaluation
            })
        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            continue
    
    print(f"Examples processed successfully: {len(processed_data)}")
    
    # Verify an example
    print("\n" + "="*60)
    print("FORMATTED TEXT EXAMPLE:")
    print("="*60)
    print(processed_data[0]["text"][:800])
    print("...")
    print("="*60 + "\n")
    
    # Split train/eval
    split_idx = int(len(processed_data) * train_split)
    train_data = processed_data[:split_idx]
    eval_data = processed_data[split_idx:]
    
    print(f"Split: {len(train_data)} train, {len(eval_data)} eval")
    
    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)
    
    return train_dataset, eval_dataset

# ==================== TRAINING FUNCTION ====================

def train():
    # Initialize wandb sweep
    wandb.init()
    config = wandb.config  # get sweep hyperparameters

    print("\n" + "="*60)
    print("SWEEP CONFIGURATION:")
    print("="*60)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("="*60 + "\n")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare dataset
    train_dataset, eval_dataset = prepare_dataset(
        json_path="../data/train_dataset.json",  
        system_prompt_path="../data/system_prompt.txt",  
        tokenizer=tokenizer,
        train_split=0.8
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    
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
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="wandb",
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        weight_decay=0.01,
        max_grad_norm=1.0,
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        tokenizer=tokenizer,
    )
    
    # Train
    print("\n Training is about to start!...\n")
    trainer.train()
    
    # Custom evaluation for tool calling accuracy
    print("\n Evaluating tool calling accuracy...")
    eval_results = evaluate_tool_calling_accuracy(
        trainer.model, 
        eval_dataset,
        tokenizer
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