#!/bin/bash

# Full evaluation script for both 3B and 7B models

echo "================================"
echo "HoliSafe-Bench Full Evaluation"
echo "Starting evaluation of Qwen-2.5-VL models"
echo "================================"
echo ""

# Create results directory
mkdir -p results

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

# Evaluate both models
echo "This will evaluate both 3B and 7B models on the full HoliSafe-Bench dataset"
echo "Expected runtime: ~8 hours on A100 GPU"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Evaluation cancelled"
    exit 0
fi

# Run evaluation
python3 evaluate.py \
    --model_size both \
    --output_dir ./results \
    --save_responses \
    --use_flash_attention

# Generate visualizations
echo ""
echo "Generating visualizations..."
python3 visualize.py --model_name qwen25vl_3b --results_dir ./results
python3 visualize.py --model_name qwen25vl_7b --results_dir ./results

echo ""
echo "================================"
echo "Evaluation Complete!"
echo "Results saved in ./results/"
echo "================================"
