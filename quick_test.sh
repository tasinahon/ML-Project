#!/bin/bash

# Quick test script for HoliSafe-Bench evaluation
# Tests on a small subset before running full evaluation

echo "================================"
echo "HoliSafe-Bench Quick Test"
echo "================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Activate venv if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Install dependencies if needed
echo "Checking dependencies..."
pip install -q -r requirements.txt

echo ""
echo "Testing on 5 samples with Qwen-2.5-VL-3B..."
echo ""

python3 evaluate.py \
    --model_size 3B \
    --max_samples 5 \
    --output_dir ./test_results \
    --save_responses

echo ""
echo "================================"
echo "Test complete!"
echo "Check ./test_results/ for outputs"
echo "================================"
echo ""
echo "To run full evaluation:"
echo "  3B model:  python3 evaluate.py --model_size 3B"
echo "  7B model:  python3 evaluate.py --model_size 7B"
echo "  Both:      python3 evaluate.py --model_size both"
