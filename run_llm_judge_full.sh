#!/bin/bash

# Full LLM-as-Judge Evaluation Script
# Evaluates all model variants using Gemma-3-4b-it as judge

echo "=========================================="
echo "Full LLM-as-Judge Evaluation Pipeline"
echo "=========================================="

# Activate virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Set judge model (Gemma 3 - multimodal, can see images!)
JUDGE_MODEL="google/gemma-3-4b-it"
OUTPUT_DIR="results/llm_judge_full"

echo "Using ${JUDGE_MODEL} as LLM judge (multimodal - analyzes images)"

# Create output directory
mkdir -p $OUTPUT_DIR

# Evaluate FP16 7B model
echo ""
echo "1. Evaluating Qwen2.5-VL 7B FP16..."
python llm_judge_eval.py \
    --response_file results/responses_qwen25vl_7b_fp16.json \
    --judge_model $JUDGE_MODEL \
    --output_dir $OUTPUT_DIR \
    --device auto

# Evaluate 4-bit quantized 7B model
echo ""
echo "2. Evaluating Qwen2.5-VL 7B 4-bit Quantized..."
python llm_judge_eval.py \
    --response_file results/responses_qwen25vl_7b_bitsandbytes4bit.json \
    --judge_model $JUDGE_MODEL \
    --output_dir $OUTPUT_DIR \
    --device auto

# Optional: Evaluate WANDA pruned model if exists
if [ -f "results/responses_qwen25vl_7b_fp16_wanda50.json" ]; then
    echo ""
    echo "3. Evaluating Qwen2.5-VL 7B WANDA 50% pruned..."
    python llm_judge_eval.py \
        --response_file results/responses_qwen25vl_7b_fp16_wanda50.json \
        --judge_model $JUDGE_MODEL \
        --output_dir $OUTPUT_DIR \
        --device auto
fi

# Optional: Evaluate 3B model if exists
if [ -f "results/responses_qwen25vl_3b.json" ]; then
    echo ""
    echo "4. Evaluating Qwen2.5-VL 3B..."
    python llm_judge_eval.py \
        --response_file results/responses_qwen25vl_3b.json \
        --judge_model $JUDGE_MODEL \
        --output_dir $OUTPUT_DIR \
        --device auto
fi

echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
