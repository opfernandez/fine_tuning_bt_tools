# Fine-Tuning LLMs for Behavior Tree Tool Calling

This project provides a framework for fine-tuning Large Language Models (LLMs) to generate tool calls that execute Behavior Trees (BTs). It uses LoRA (Low-Rank Adaptation) for efficient fine-tuning and Weights & Biases (W&B) for hyperparameter sweeps and experiment tracking.

## Overview

The goal is to train models that can:
1. Understand natural language commands
2. Select the appropriate Behavior Tree to execute
3. Generate correctly formatted tool calls with proper arguments

The trained model outputs structured tool calls in the format:
```json
{
  "name": "execute_behavior_tree",
  "arguments": {
    "bt_xml_filename": "<behavior_tree>.xml",
    "execution_id": "<agent_id>",
    "input_parameters": { "param1": "value1" }
  }
}
```

## Project Structure

```
fine_tuning_bt_tools/
├── adapters/              # Saved LoRA adapters from training runs
├── checkpoints/           # Training checkpoints
├── configs/
│   └── sweep_config.yaml  # W&B sweep configuration
├── convert_model_to_gguf/
│   └── merge_model.py     # Merge LoRA adapters with base model
├── data/
│   ├── train_dataset.json # Training data in ChatML format
│   ├── system_prompt.txt  # System prompt with BT descriptions
│   └── test_data.json     # Test dataset
├── notebooks/
│   └── try_adapters.ipynb # Notebook for testing trained adapters
├── templates/
│   ├── qwen3.jinja        # Chat template for Qwen models
│   └── mistral.jinja      # Chat template for Mistral models
└── training/
    ├── fine_tuning.py     # Main training script with W&B integration
    ├── data_loader.py     # Dataset preparation and collation
    ├── eval.py            # Tool calling evaluation metrics
    └── .env               # Environment variables (HF_TOKEN, WANDB_*)
```

## Installation

### Requirements

```bash
pip install torch transformers peft datasets wandb python-dotenv
```

### Environment Setup

Create a `.env` file in the `training/` directory:

```bash
cp training/.env.example training/.env
```

Edit the `.env` file with your credentials:

```
HF_TOKEN=your_hugging_face_token_here
WANDB_ENTITY=your_wandb_entity_here
WANDB_PROJECT=your_wandb_project_here
```

## Data Format

The training data follows the ChatML format with tool calls. Each example is a conversation with system prompt, user messages, assistant tool calls, and tool responses:

```json
{
  "train": [
    {
      "messages": [
        {
          "role": "system",
          "content": "You are a robot control assistant."
        },
        {
          "role": "user",
          "content": "Your assigned agent ID is 1. And your task is: Navigate to the kitchen."
        },
        {
          "role": "assistant",
          "content": "",
          "tool_calls": [
            {
              "name": "execute_behavior_tree",
              "arguments": {
                "bt_xml_filename": "move.xml",
                "execution_id": "1",
                "input_parameters": {
                  "region": "kitchen"
                }
              }
            }
          ]
        },
        {
          "role": "tool",
          "content": "The execution of the Behavior Tree with filename move.xml (ID: 1) has been successful."
        },
        {
          "role": "assistant",
          "content": "Successfully navigated to the kitchen."
        }
      ]
    }
  ]
}
```

## W&B Sweeps

### Sweep Configuration

The sweep configuration file (`configs/sweep_config.yaml`) defines the hyperparameter search space:

```yaml
program: fine_tuning.py
method: random  # Options: bayes, random, grid
metric:
  name: eval/loss
  goal: minimize

name: qwen-toolcalling-lora

parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 1e-5
    max: 1e-3
  
  lora_r:
    values: [8, 16, 32]
  
  lora_alpha:
    values: [16, 32, 64]
  
  num_train_epochs:
    values: [1, 2, 3]
  
  gradient_accumulation_steps:
    values: [8, 16, 32, 64]
  
  lora_dropout:
    distribution: uniform
    min: 0.0
    max: 0.1
  
  warmup_ratio:
    values: [0.05, 0.1, 0.15]

run_cap: 10  # Maximum number of runs
```

### Running Sweeps

1. **Create a new sweep:**
   ```bash
   cd training
   wandb sweep ../configs/sweep_config.yaml
   ```
   This will output a sweep ID like `username/project/abc123`.

2. **Start sweep agents:**
   ```bash
   # Run a single agent
   wandb agent username/project/abc123
   
   # Run multiple agents in parallel (different terminals)
   wandb agent username/project/abc123 --count 5
   ```

3. **Resume an existing sweep:**
   ```bash
   wandb agent username/project/abc123
   ```

4. **Monitor sweeps:**
   - Go to your W&B dashboard: `https://wandb.ai/username/project`
   - View real-time metrics, compare runs, and analyze results

### Common Commands

```bash
# Create sweep and get sweep ID
wandb sweep configs/sweep_config.yaml

# Start agent with specific sweep
wandb agent username/project/sweep_id

# Run single training (without sweep)
cd training
python fine_tuning.py

# List all sweeps in project
wandb sweep --list

# Stop a sweep
wandb sweep --stop username/project/sweep_id
```

## Training Details

### LoRA Configuration

The training uses LoRA to efficiently fine-tune the model by targeting attention and MLP layers:

```python
lora_config = LoraConfig(
    r=config.lora_r,              # Rank (8, 16, or 32)
    lora_alpha=config.lora_alpha, # Scaling factor
    lora_dropout=config.lora_dropout,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                   "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)
```

### Training Output

After each run, the following are saved:
- **Checkpoints:** `checkpoints/{run_name}/` - intermediate checkpoints
- **Best Adapter:** `adapters/{run_name}_best/` - final LoRA adapter

Run naming convention: `r{lora_r}_a{lora_alpha}_lr{learning_rate}_bs{batch_size}`

Example: `r32_a64_lr2e-04_bs8_best`

## Evaluation Metrics

The evaluation module (`training/eval.py`) computes three metrics:

| Metric | Description |
|--------|-------------|
| `tool_name_acc` | Percentage of correct tool name predictions |
| `arg_exact` | Percentage of exact argument matches |
| `valid_json` | Percentage of valid JSON outputs |

These metrics are logged to W&B under `final/tool_name_accuracy`, `final/arg_exact_match`, and `final/valid_json_rate`.

## Using Trained Adapters

### Testing with Notebook

Use `notebooks/try_adapters.ipynb` to interactively test trained adapters:

1. Load the model and processor
2. Prepare the evaluation dataset
3. Run inference and compare with expected outputs
4. Evaluate tool calling accuracy

## Converting to GGUF

To convert a trained adapter to GGUF format for use with llama.cpp:

```bash
cd convert_model_to_gguf
python merge_model.py --adapter_path ../adapters/r32_a64_lr2e-04_bs8_best
```

