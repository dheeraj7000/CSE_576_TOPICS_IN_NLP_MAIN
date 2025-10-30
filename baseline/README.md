# LogiQA Baseline Evaluation

This directory contains baseline evaluation scripts for the LogiQA benchmark using Llama-3.2-3B model.

## Features

- **Simple Inference**: Basic model evaluation without optimization
- **FSDP Support**: Fully Sharded Data Parallel for memory efficiency
- **DeepSpeed Integration**: Advanced memory optimization and parallelization
- **Comprehensive Results**: Detailed JSON output with per-example analysis

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set HuggingFace Token (if needed)
```bash
export HF_TOKEN="your_token_here"
```

### 3. Run Evaluation

#### Simple Inference (Default)
```bash
./run_baseline.sh simple
```

#### With FSDP
```bash
./run_baseline.sh fsdp
```

#### With DeepSpeed
```bash
./run_baseline.sh deepspeed
```

#### For 4GB GPU (Optimized)
```bash
./run_baseline.sh 4gb 50  # Memory-optimized version
```

#### Limit Samples (for testing)
```bash
./run_baseline.sh simple 100  # Evaluate only 100 samples
```

## Manual Usage

### Basic Command
```bash
python logiqa_baseline.py --model_name meta-llama/Llama-3.2-3B
```

### With FSDP
```bash
python logiqa_baseline.py --model_name meta-llama/Llama-3.2-3B --use_fsdp
```

### With DeepSpeed
```bash
deepspeed logiqa_baseline.py --model_name meta-llama/Llama-3.2-3B --use_deepspeed
```

## Arguments

### Model Arguments
- `--model_name`: Model name or path (default: meta-llama/Llama-3.2-3B)

### Optimization Arguments
- `--use_fsdp`: Enable FSDP parallelization
- `--use_deepspeed`: Enable DeepSpeed optimization
- `--deepspeed_config`: DeepSpeed config file (default: deepspeed_config.json)

### Dataset Arguments
- `--dataset_path`: Local dataset path (fallback)
- `--max_samples`: Limit evaluation samples

### Generation Arguments
- `--max_length`: Maximum input length (default: 2048)
- `--max_new_tokens`: Maximum new tokens (default: 10)
- `--temperature`: Generation temperature (default: 0.1)
- `--do_sample`: Enable sampling

### Output Arguments
- `--output_dir`: Results directory (default: ./baseline_results)

## Output Files

The evaluation generates:

1. **Detailed Results**: `logiqa_results_Llama-3.2-3B.json`
   - Per-example predictions and correctness
   - Model responses and extracted answers
   - Configuration parameters

2. **Summary**: `summary.txt`
   - Overall accuracy
   - Total examples processed
   - Optimization mode used

## Example Output Structure

```json
{
  "model_name": "meta-llama/Llama-3.2-3B",
  "total_examples": 651,
  "correct_predictions": 234,
  "accuracy": 0.3594,
  "optimization": {
    "use_fsdp": false,
    "use_deepspeed": false,
    "simple_inference": true
  },
  "results": [
    {
      "example_id": 0,
      "context": "...",
      "question": "...",
      "options": ["A", "B", "C", "D"],
      "correct_answer": "B",
      "predicted_answer": "A",
      "response": "A",
      "is_correct": false
    }
  ]
}
```

## Memory Requirements

- **Simple Inference**: ~12GB GPU memory
- **FSDP**: ~8GB GPU memory (distributed)
- **DeepSpeed**: ~6GB GPU memory (with ZeRO stage 2)
- **4GB Optimized**: ~3.5GB GPU memory (ultra-efficient)

## Troubleshooting

1. **CUDA Out of Memory**: Use FSDP or DeepSpeed
2. **Dataset Loading Issues**: Check internet connection or provide local dataset
3. **Model Access**: Ensure HuggingFace token is set for gated models