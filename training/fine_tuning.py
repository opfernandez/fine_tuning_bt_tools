import torch
import wandb
import dotenv
import json
import os
import shutil
from transformers import (Trainer, TrainingArguments, AutoModelForCausalLM,
    AutoProcessor, TrainerCallback, TrainerControl, TrainerState)
from peft import LoraConfig, get_peft_model
from eval import evaluate_tool_calling_accuracy
from data_loader import DataCollatorForChatML, prepare_dataset

dotenv.load_dotenv()  # Load environment variables from .env file
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN") 
# model_id = "Qwen/Qwen3-0.6B"
model_id = "unsloth/functiongemma-270m-it"
model_type = "functiongemma"  # Change to functiongemma or qwen3 

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
        """
        Args:
            patience:  Number of evaluations to wait for improvement before stopping.
            threshold: Minimum relative improvement in eval loss to reset patience (e.g., 0.05 for 5% improvement).
        """
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

        # Use relative improvement: current must be at least (1 - threshold) * best_loss
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
    # Initialize wandb sweep
    wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project=os.getenv("WANDB_PROJECT"),
        mode="online", # Change to "online" when you want to log to the cloud
    )
    config = wandb.config  # get sweep hyperparameters

    # Create a unique run name based on hyperparameters for better tracking
    run_name = (
        f"ep{config.num_train_epochs}"
        f"_r{config.lora_r}"
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
    with open(os.path.join("../templates", "functiongemma.jinja"), "r", encoding="utf-8") as f:
        custom_template = f.read()
    processor.chat_template = custom_template
    
    # Prepare dataset
    train_dataset, eval_dataset, raw_eval_data = prepare_dataset(
        json_path=os.path.join("../data", "domotic_dataset.json"),  
        system_prompt_path=os.path.join("../data", "system_prompt_domotic.jinja"),  
        processor=processor,
        tools=bt_tool,
        train_split=0.8,
        max_length=1024
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
    model.enable_input_require_grads()
    
    output_dir = os.path.join("../checkpoints", wandb.run.name)
    # Training arguments with sweep hyperparameters
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=1,  # Keep this fixed to lower value and use gradient accumulation so vram does not explode
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
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
        gradient_checkpointing=True,  # Enabled to save VRAM (trades compute for memory)
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
        max_length=1024, # Inspect some examples to set this appropriately based on your data and model context window
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
        callbacks=[EarlyStoppingCallback(patience=2, threshold=0.05)]
    )
    
    # Train
    print("\n Training is about to start!...\n")
    trainer.train()
    
    # Custom evaluation for tool calling accuracy
    # Check if early stopping was triggered and if loss is below threshold
    early_stopping_callback = None
    for callback in trainer.callback_handler.callbacks:
        if isinstance(callback, EarlyStoppingCallback):
            early_stopping_callback = callback
            break
    
    should_evaluate = True
    eval_loss_threshold = 0.35  # Threshold to decide whether to perform final evaluation
    
    if early_stopping_callback:
        print(f"\n Training stopped early with best loss: {early_stopping_callback.best_loss:.4f}")
        if early_stopping_callback.best_loss > eval_loss_threshold:
            print(f" Loss {early_stopping_callback.best_loss:.4f} is above threshold {eval_loss_threshold:.4f}, skipping final evaluation")
            should_evaluate = False
        else:
            print(f" Loss {early_stopping_callback.best_loss:.4f} is below threshold {eval_loss_threshold:.4f}, proceeding with final evaluation")
    
    if should_evaluate:
        # print("\n Evaluating tool calling accuracy...")
        eval_results = evaluate_tool_calling_accuracy(
            trainer.model, 
            raw_eval_data,
            processor,
            tools=bt_tool,
            model_type=model_type
        )
        
        # Log custom metrics
        wandb.log({
            "final/tool_name_accuracy": eval_results["tool_name_acc"],
            "final/arg_exact_match": eval_results["arg_exact"],
            "final/valid_json_rate": eval_results["valid_json"]
        })
    
    # Save the best model
    best_model_path = os.path.join("../adapters", f"{model_type}_{wandb.run.name}_best")
    trainer.save_model(best_model_path)
    print(f"\n Best model saved at: {best_model_path}")
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