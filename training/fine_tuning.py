import torch
import wandb
import dotenv
import os
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoProcessor
from peft import LoraConfig, get_peft_model
from eval import evaluate_tool_calling_accuracy
from data_loader import DataCollatorForChatML, prepare_dataset

dotenv.load_dotenv()  # Load environment variables from .env file
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN") 
model_id = "Qwen/Qwen3-0.6B"


# Define tool schema to be parsed in the chat template
bt_tool = [{
            "name": "execute_behavior_tree",
            "arguments": {
                "bt_xml_filename": "<selected_bt>.xml",
                "execution_id": "<your numeric agent id>",
                "input_parameters": { "arg1": "value1" }
            }
        }]

# ==================== TRAINING FUNCTION ====================

def train():
    """Main training function with wandb sweep integration"""
    # Initialize wandb sweep
    wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project=os.getenv("WANDB_PROJECT"),
        mode="online", # Change to "online" when you want to log to the cloud
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
        tools=bt_tool,
        train_split=0.8
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        # attn_implementation="flash_attention_2"
    )
    # print(f"\nModel loaded: {model}")
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
        eval_steps=5,
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
    # print("\n Setting up trainer...")
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
    # print("\n Evaluating tool calling accuracy...")
    eval_results = evaluate_tool_calling_accuracy(
        trainer.model, 
        raw_eval_data,
        processor,
        tools=bt_tool
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