from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model = "Qwen/Qwen3-0.6B"
lora_checkpoint = "r32_a32_lr2e-04_bs8_best"
lora_path = "../adapters/" + lora_checkpoint

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(model, lora_path)

tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.save_pretrained(f"./merged_models/{lora_checkpoint}_merged")

# Merge
model = model.merge_and_unload()

model.save_pretrained(f"./merged_models/{lora_checkpoint}_merged")