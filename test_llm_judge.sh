#!/bin/bash

# Test LLM-as-Judge evaluation on a small sample
# This script tests the evaluation pipeline before running on the full dataset

echo "=========================================="
echo "Testing LLM Judge Evaluation"
echo "=========================================="

# Activate virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Test with 10 samples first
echo ""
echo "Testing with 10 samples from FP16 7B model..."
echo "Using google/gemma-3-4b-it as judge (multimodal - can see images!)"
python llm_judge_eval.py \
    --response_file results/responses_qwen25vl_7b_fp16.json \
    --max_samples 10 \
    --judge_model google/gemma-3-4b-it \
    --output_dir results/llm_judge_test \
    --device auto

echo ""
echo "=========================================="
echo "Test complete! Check results/llm_judge_test/ for output"
echo "=========================================="
