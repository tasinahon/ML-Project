# SafeQwen2.5-VL Evaluation with Gemma Judge

Evaluation framework for SafeQwen2.5-VL comparing full precision (FP16) vs quantized (4-bit, 8-bit) with memory management and Gemma as LLM judge.

## Features

- ✅ **SafeQwen Support**: Built-in safety classification + text generation
- ✅ **Gemma LLM Judge**: Local Gemma model for refusal detection (no API costs)
- ✅ **Memory Management**: Aggressive cleanup to prevent PC crashes
- ✅ **Quantization**: Compare FP16 vs 4-bit/8-bit quantization
- ✅ **Safety Scores**: Track 20 safety categories from SafeQwen's classifier

## Quick Start (10 samples test)

```bash
chmod +x test_safeqwen.sh
./test_safeqwen.sh
```

This will:
1. Test FP16 on 10 samples (~5 min)
2. Test 4-bit quantization on 10 samples (~5 min)  
3. Compare results

## Full Evaluation

### Step 1: FP16 Baseline

```bash
# Run in background (uses ~16GB VRAM)
nohup python evaluate_safeqwen.py \
    --model_size 7B \
    --max_samples 4031 \
    --use_gemma_judge \
    --enable_safety_classifier \
    --memory_cleanup_interval 10 \
    > logs/safeqwen_fp16.log 2>&1 &

# Monitor progress
tail -f logs/safeqwen_fp16.log
```

### Step 2: 4-bit Quantization

```bash
# Run in background (uses ~4GB VRAM)
nohup python evaluate_safeqwen.py \
    --model_size 7B \
    --quantization bitsandbytes \
    --bits 4 \
    --max_samples 4031 \
    --use_gemma_judge \
    --enable_safety_classifier \
    --memory_cleanup_interval 10 \
    > logs/safeqwen_4bit.log 2>&1 &

# Monitor progress
tail -f logs/safeqwen_4bit.log
```

### Step 3: Compare Results

```bash
python compare_safeqwen.py --model_size 7b
```

## Output Files

```
safeqwen_results/
├── responses_safeqwen_7b_fp16.json          # FP16 responses + safety scores
├── metrics_safeqwen_7b_fp16.json            # FP16 metrics
├── config_safeqwen_7b_fp16.json             # FP16 config
├── responses_safeqwen_7b_bitsandbytes4bit.json  # 4-bit responses
├── metrics_safeqwen_7b_bitsandbytes4bit.json    # 4-bit metrics
└── config_safeqwen_7b_bitsandbytes4bit.json     # 4-bit config
```

## Metrics Tracked

### 1. Pattern-based Detection
- Regex patterns for refusal keywords
- Fast, no extra compute

### 2. Gemma Judge
- Local Gemma-2-2b-it model
- More accurate than patterns
- No API costs

### 3. Safety Classifier (SafeQwen only)
- 20 safety categories
- Built into SafeQwen model
- Probabilities for each category

## Memory Management

The framework includes aggressive memory management:

```python
--memory_cleanup_interval 10  # Clean CUDA cache every 10 samples
```

**Recommendations**:
- **16GB VRAM**: Use interval of 10
- **12GB VRAM**: Use interval of 5
- **8GB VRAM**: Use 4-bit quantization + interval of 3

## Command Line Options

```bash
python evaluate_safeqwen.py \
    --model_size 7B \                        # Model size
    --quantization bitsandbytes \            # None for FP16
    --bits 4 \                               # 4 or 8
    --max_samples 100 \                      # Number of samples
    --use_gemma_judge \                      # Enable Gemma judge
    --gemma_model google/gemma-2-2b-it \     # Gemma model
    --enable_safety_classifier \             # Use SafeQwen's classifier
    --memory_cleanup_interval 10 \           # Cleanup frequency
    --output_dir ./safeqwen_results          # Output directory
```

## Expected Results

Based on preliminary tests, SafeQwen should show:

**Hypothesis**: SafeQwen is heavily safety-tuned, so quantization may have larger impact

| Method | Expected ASR | Expected Impact |
|--------|-------------|-----------------|
| FP16 baseline | 5-15% | Baseline (very safe) |
| 4-bit quantized | 10-25% | +5-10pp degradation |
| 8-bit quantized | 7-20% | +2-5pp degradation |

*Compare with regular Qwen2.5-VL (~67% ASR)*

## Troubleshooting

### Out of Memory

```bash
# Use 4-bit quantization
python evaluate_safeqwen.py --quantization bitsandbytes --bits 4

# Increase cleanup frequency
python evaluate_safeqwen.py --memory_cleanup_interval 5

# Test with fewer samples first
python evaluate_safeqwen.py --max_samples 50
```

### PC Freezing/Overheating

```bash
# Add delays between samples
python evaluate_safeqwen.py --batch_size 1 --memory_cleanup_interval 5

# Monitor GPU temp
watch -n 1 nvidia-smi

# If temp > 80°C, stop and cool down
pkill -9 python
```

### Gemma Loading Issues

```bash
# Use smaller Gemma model
python evaluate_safeqwen.py --gemma_model google/gemma-2b-it

# Disable Gemma (pattern matching only)
python evaluate_safeqwen.py  # Don't use --use_gemma_judge flag
```

## Comparison with Regular Qwen

After running SafeQwen evaluation, compare with regular Qwen results:

```bash
# Regular Qwen results are in ./results/
# SafeQwen results are in ./safeqwen_results/

# Compare FP16 versions
echo "Regular Qwen ASR: $(cat results/metrics_qwen25vl_7b_fp16.json | jq .attack_success_rate)"
echo "SafeQwen ASR: $(cat safeqwen_results/metrics_safeqwen_7b_fp16.json | jq .attack_success_rate)"
```

## Safety Categories (SafeQwen Classifier)

When `--enable_safety_classifier` is used, each response includes probabilities for:

1. weapon_related_violence
2. sexual_content
3. hate_speech
4. illegal_activities  
5. self_harm
6. ... (15 more categories)

Check `responses_*.json` files for per-sample safety scores.

## Performance

| Task | Samples | Time (7B FP16) | VRAM |
|------|---------|----------------|------|
| SafeQwen generation | 10 | ~2 min | ~16GB |
| Gemma judging | 10 | ~1 min | ~2GB |
| **Total pipeline** | 10 | ~3 min | ~16GB peak |
| **Full evaluation** | 4031 | ~6-8 hours | ~16GB peak |

*Times on RTX 3090. Quantized versions are similar speed but use less VRAM.*

## Citation

```bibtex
@article{safeqwen2024,
  title={SafeQwen: Safe Multimodal Large Language Model},
  author={ETRI Vision-Language Lab},
  year={2024}
}
```
