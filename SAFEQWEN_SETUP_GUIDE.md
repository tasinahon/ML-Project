# SafeQwen Evaluation - Complete Setup Guide

## Overview

This setup evaluates **SafeQwen2.5-VL-7B** (safety-tuned Qwen) with:
- ✅ **Full Precision (FP16)** baseline
- ✅ **4-bit/8-bit Quantization** for comparison
- ✅ **Gemma-2-2b-it** as LLM judge (no API costs)
- ✅ **Safety Classification** (20 categories from SafeQwen)
- ✅ **Memory Management** to prevent PC crashes

## Why SafeQwen?

SafeQwen is specifically safety-tuned, so comparing it to regular Qwen tests:
- **Hypothesis**: Safety tuning makes models more fragile to quantization
- **Expected**: Larger ASR increase than regular Qwen (1.8pp → 5-15pp?)

## Installation (One-time)

```bash
cd /home/user/Downloads/minigo/2005057

# Activate venv (if not already active)
source venv/bin/activate

# Install required packages
pip install transformers datasets pillow qwen-vl-utils
pip install torch torchvision accelerate
pip install bitsandbytes  # For quantization
pip install pandas matplotlib seaborn  # For comparison

# Test setup
python test_setup.py
```

Expected output: "✅ ALL TESTS PASSED!"

## Quick Test (10 minutes)

```bash
./test_safeqwen.sh
```

This runs:
1. **FP16** on 10 samples (~5 min, ~16GB VRAM)
2. **4-bit** on 10 samples (~5 min, ~4GB VRAM)
3. **Comparison** report

## Full Evaluation (6-8 hours)

### Step 1: FP16 Baseline

```bash
# Create logs directory
mkdir -p logs

# Run FP16 evaluation
nohup python evaluate_safeqwen.py \
    --model_size 7B \
    --max_samples 4031 \
    --use_gemma_judge \
    --enable_safety_classifier \
    --memory_cleanup_interval 10 \
    --output_dir ./safeqwen_results \
    > logs/safeqwen_fp16.log 2>&1 &

# Get process ID
echo $! > logs/safeqwen_fp16.pid

# Monitor progress
tail -f logs/safeqwen_fp16.log

# Check if still running
ps aux | grep evaluate_safeqwen

# Watch GPU usage
watch -n 2 nvidia-smi
```

**Expected**:
- Runtime: ~6-8 hours
- VRAM: ~16GB (SafeQwen) + ~2GB (Gemma judge)
- Peak: ~18GB during model switching

### Step 2: Wait for Completion

```bash
# Check if done
tail -20 logs/safeqwen_fp16.log

# Should see "Evaluation complete!" at the end
```

### Step 3: 4-bit Quantization

```bash
# Run 4-bit evaluation
nohup python evaluate_safeqwen.py \
    --model_size 7B \
    --quantization bitsandbytes \
    --bits 4 \
    --max_samples 4031 \
    --use_gemma_judge \
    --enable_safety_classifier \
    --memory_cleanup_interval 10 \
    --output_dir ./safeqwen_results \
    > logs/safeqwen_4bit.log 2>&1 &

echo $! > logs/safeqwen_4bit.pid

# Monitor
tail -f logs/safeqwen_4bit.log
```

**Expected**:
- Runtime: ~6-8 hours (similar to FP16)
- VRAM: ~4-5GB (SafeQwen quantized) + ~2GB (Gemma)
- Peak: ~6-7GB

### Step 4: Compare Results

```bash
python compare_safeqwen.py --model_size 7b --results_dir ./safeqwen_results
```

## Memory Safety Tips

### Monitor System

```bash
# GPU monitoring (Terminal 1)
watch -n 1 nvidia-smi

# System resources (Terminal 2)
htop

# Log monitoring (Terminal 3)
tail -f logs/safeqwen_*.log
```

### If PC Starts Lagging

```bash
# Check GPU temp
nvidia-smi --query-gpu=temperature.gpu --format=csv

# If temp > 80°C, stop immediately
pkill -9 python

# Let cool down for 10-15 minutes
```

### Recovery from Crash

```bash
# Clear all GPU memory
nvidia-smi --gpu-reset

# Or reboot if needed
sudo reboot

# Resume from where it stopped (results are saved incrementally)
# Check last saved sample
ls -lt safeqwen_results/

# Restart with remaining samples
python evaluate_safeqwen.py --max_samples REMAINING --skip COMPLETED
```

## Expected Timeline

| Task | Time | VRAM | Notes |
|------|------|------|-------|
| Setup test | 5 min | 16GB | Test 1 sample |
| Quick test | 10 min | 16GB | Test 10 samples |
| FP16 full | 6-8h | 18GB | 4031 samples |
| 4-bit full | 6-8h | 7GB | 4031 samples |
| 8-bit full | 6-8h | 10GB | Optional |

**Total**: ~14-16 hours for full FP16 + 4-bit comparison

## Understanding Results

### Output Files

```
safeqwen_results/
├── metrics_safeqwen_7b_fp16.json          # Main results
├── responses_safeqwen_7b_fp16.json        # All responses + safety scores
└── config_safeqwen_7b_fp16.json           # Settings used
```

### Metrics Explained

```json
{
  "refusal_rate": 32.5,              // Pattern-based detection
  "attack_success_rate": 67.5,        // Pattern-based ASR
  "gemma_refusal_rate": 35.2,         // Gemma judge (more accurate)
  "gemma_asr": 64.8,                  // Gemma judge ASR
  "safety_classifier_rate": 89.3,     // % with safety flags
  "category_metrics": { ... }         // Per-category breakdown
}
```

### Comparison Interpretation

When you run `compare_safeqwen.py`, look for:

1. **ASR Delta**: How much did quantization increase attacks?
   - < 2pp: Minimal impact (like regular Qwen)
   - 2-5pp: Moderate impact
   - > 5pp: **Significant safety degradation** ← Expected for SafeQwen!

2. **Category Fragility**: Which categories degraded most?
   - weapon_related_violence
   - illegal_activities
   - sexual_content

3. **Safety Classifier**: Did it still detect unsafe content?
   - If yes: Model is unsafe but knows it
   - If no: Model is unsafe AND doesn't know it (worse!)

## Comparison with Regular Qwen

After completing SafeQwen evaluation:

```bash
# Compare SafeQwen vs Regular Qwen (both FP16)
echo "Regular Qwen ASR:"
jq .attack_success_rate results/metrics_qwen25vl_7b_fp16.json

echo "SafeQwen ASR:"
jq .gemma_asr safeqwen_results/metrics_safeqwen_7b_fp16.json

# Expected: SafeQwen much lower (~10-20% vs ~67%)
```

## Troubleshooting

### Issue: OOM during FP16

**Solution**: Use 4-bit from the start
```bash
python evaluate_safeqwen.py --quantization bitsandbytes --bits 4
```

### Issue: PC freezing

**Solution**: More aggressive memory cleanup
```bash
python evaluate_safeqwen.py --memory_cleanup_interval 5
```

### Issue: Slow progress

**Solution**: Disable Gemma judge temporarily
```bash
# Run without --use_gemma_judge flag
python evaluate_safeqwen.py --model_size 7B

# Then run Gemma separately on saved responses (coming soon)
```

### Issue: Gemma loading fails

**Solution**: Use smaller Gemma model
```bash
python evaluate_safeqwen.py --gemma_model google/gemma-2b
```

## Research Questions

After running evaluation, you can answer:

1. **Does safety tuning increase quantization fragility?**
   - Compare SafeQwen 4-bit delta vs Regular Qwen 4-bit delta

2. **Which safety categories are most fragile?**
   - Check category-wise comparison table

3. **Does safety classifier still work after quantization?**
   - Compare `safety_classifier_rate` between FP16 and 4-bit

4. **Is Gemma judge reliable?**
   - Compare `gemma_asr` vs `pattern_asr` (should be within 5pp)

## Next Steps

1. ✅ **Test setup** (`python test_setup.py`)
2. ✅ **Quick test** (`./test_safeqwen.sh`)
3. **Full FP16** (6-8 hours)
4. **Full 4-bit** (6-8 hours)
5. **Comparison** (`python compare_safeqwen.py`)
6. **Write paper** (use results from safeqwen_results/)

## Support

If you encounter issues:

```bash
# Check setup
python test_setup.py

# Test single sample
python evaluate_safeqwen.py --max_samples 1

# Monitor GPU
watch -n 1 nvidia-smi

# Check logs
tail -100 logs/safeqwen_*.log
```

## File Summary

| File | Purpose |
|------|---------|
| `safeqwen_wrapper.py` | SafeQwen model with safety classification |
| `gemma_judge.py` | Local Gemma LLM judge |
| `evaluate_safeqwen.py` | Main evaluation script |
| `compare_safeqwen.py` | Comparison tool |
| `test_safeqwen.sh` | Quick test script |
| `test_setup.py` | Setup verification |
| `SAFEQWEN_README.md` | User documentation |
| `THIS FILE` | Complete setup guide |

---

**Ready to start?** Run: `python test_setup.py`
