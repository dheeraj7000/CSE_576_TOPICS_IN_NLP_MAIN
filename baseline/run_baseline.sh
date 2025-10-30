#!/bin/bash

# LogiQA Baseline Evaluation Script
# Usage: ./run_baseline.sh [simple|fsdp|deepspeed] [max_samples]

set -e

# Default parameters
MODE=${1:-"simple"}
MAX_SAMPLES=${2:-""}
MODEL_NAME="meta-llama/Llama-3.2-3B"
OUTPUT_DIR="./baseline_results"

echo "Running LogiQA Baseline Evaluation"
echo "Mode: $MODE"
echo "Model: $MODEL_NAME"

# Create output directory
mkdir -p $OUTPUT_DIR

# Set common arguments
COMMON_ARGS="--model_name $MODEL_NAME --output_dir $OUTPUT_DIR"

if [ ! -z "$MAX_SAMPLES" ]; then
    COMMON_ARGS="$COMMON_ARGS --max_samples $MAX_SAMPLES"
fi

case $MODE in
    "simple")
        echo "Running simple inference..."
        python logiqa_baseline.py $COMMON_ARGS
        ;;
    "4gb")
        echo "Running 4GB GPU optimized version..."
        python logiqa_4gb.py --max_samples ${MAX_SAMPLES:-50}
        ;;
    "fsdp")
        echo "Running with FSDP..."
        python logiqa_baseline.py $COMMON_ARGS --use_fsdp
        ;;
    "deepspeed")
        echo "Running with DeepSpeed..."
        deepspeed logiqa_baseline.py $COMMON_ARGS --use_deepspeed --deepspeed_config deepspeed_config.json
        ;;
    *)
        echo "Invalid mode: $MODE"
        echo "Usage: $0 [simple|fsdp|deepspeed|4gb] [max_samples]"
        echo "  4gb: Memory-optimized for 4GB GPU"
        exit 1
        ;;
esac

echo "Evaluation completed. Results saved to $OUTPUT_DIR"