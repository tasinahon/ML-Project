# Quantization Safety Research Guide

## Project: Quantization-Induced Safety Drift in Multimodal LLMs

This guide shows how to use the evaluation framework for your quantization safety research.

---

## 🎯 Research Workflow

### Phase 1: Baseline (FP16) Evaluation

**Currently Running:**
```bash
# 7B FP16 baseline (in progress via nohup)
tail -f eval_7b.log
```

**Or start fresh:**
```bash
source venv/bin/activate
python evaluate.py --model_size 7B --output_dir ./results
```

**Output:** `results/metrics_qwen25vl_7b_fp16.json`

---

### Phase 2: Quantized Model Evaluation

#### 4-bit Quantization (BitsAndBytes)
```bash
source venv/bin/activate
python evaluate.py \
    --model_size 7B \
    --quantization bitsandbytes \
    --bits 4 \
    --output_dir ./results
```

**Output:** `results/metrics_qwen25vl_7b_bitsandbytes4bit.json`

#### 8-bit Quantization
```bash
python evaluate.py \
    --model_size 7B \
    --quantization bitsandbytes \
    --bits 8 \
    --output_dir ./results
```

**Output:** `results/metrics_qwen25vl_7b_bitsandbytes8bit.json`

---

### Phase 3: Compare & Analyze Safety Drift

```bash
python compare_quantization.py \
    --model_size 7b \
    --baseline fp16 \
    --quantized bitsandbytes4bit \
    --results_dir ./results
```

**Generates:**
- Console report with key findings
- `safety_drift_7b.png` - Overall RR/ASR comparison
- `category_drift_7b.png` - Category-specific degradation

---

## 📊 Expected Results (Hypothesis Testing)

### H1: Fragile Guardrail Hypothesis
**Prediction:** Some categories will show larger ASR increase than others

**Measurement:**
- Compare category-wise Δ ASR
- Identify which safety types are most fragile

**Example output:**
```
MOST FRAGILE SAFETY CATEGORIES:
   violence: ASR +15.3% (RR -12.1%)
   illegal_activity: ASR +12.7% (RR -8.4%)
   specialized_advice: ASR +5.2% (RR -3.1%)
```

### H2: Mechanism Drift
**Prediction:** Quantization degrades primary refusal circuits

**Measurement:**
- Overall Δ ASR > 0 indicates degradation
- Category-specific patterns reveal which circuits fail first

---

## 🔬 Research Timeline

| Task | Command | Time | Output |
|------|---------|------|--------|
| **1. FP16 Baseline** | `evaluate.py --model_size 7B` | ~6h | `*_fp16.json` |
| **2. 4-bit Quant** | `evaluate.py --quantization bitsandbytes --bits 4` | ~4h | `*_4bit.json` |
| **3. 8-bit Quant** | `evaluate.py --quantization bitsandbytes --bits 8` | ~5h | `*_8bit.json` |
| **4. Analysis** | `compare_quantization.py` | <1m | Report + plots |

**Total:** ~15 hours for complete comparison

---

## 💾 Memory Requirements

| Configuration | VRAM | Speed |
|--------------|------|-------|
| 7B FP16 | ~16GB | Baseline |
| 7B 8-bit | ~8GB | 0.8-0.9x |
| 7B 4-bit | ~4GB | 0.6-0.7x |

---

## 📈 Quick Progress Check

**Check if evaluation is running:**
```bash
ps aux | grep evaluate.py
```

**See progress:**
```bash
# For FP16 run
tail -100 eval_7b.log | grep "Generating with"

# Count completed samples
grep -c '"response":' results/responses_qwen25vl_7b_fp16.json
```

**Estimate time remaining:**
```bash
# If 500 samples done out of 4031
# Remaining = (4031 - 500) * 0.5sec/sample = ~30 min
```

---

## 🎓 For Your Paper

### Key Metrics to Report

1. **Overall Safety Drift:**
   - Δ ASR (Attack Success Rate change)
   - Δ RR (Refusal Rate change)

2. **Category-Specific Fragility:**
   - Which categories degrade most?
   - Evidence for "sparse safety neurons" (H1)

3. **Quantization Comparison:**
   - FP16 vs 8-bit vs 4-bit
   - When does safety critically degrade?

### Sample Results Table (for paper)

```
Model      | RR (%)  | ASR (%) | Δ ASR
-----------|---------|---------|-------
7B-FP16    | 0.6     | 52.5    | -
7B-INT8    | 0.4     | 58.3    | +5.8
7B-INT4    | 0.2     | 67.1    | +14.6
```

---

## 🚀 Next Steps After Baseline

Once `eval_7b.log` shows completion:

1. **Verify results match paper:**
   ```bash
   python compare_quantization.py --model_size 7b --baseline fp16 --quantized fp16
   # Should show ~52.5% ASR if code is correct
   ```

2. **Run quantized evaluation:**
   ```bash
   nohup python evaluate.py --model_size 7B --quantization bitsandbytes --bits 4 --output_dir ./results > eval_7b_4bit.log 2>&1 &
   ```

3. **Compare and analyze:**
   ```bash
   python compare_quantization.py --model_size 7b --baseline fp16 --quantized bitsandbytes4bit
   ```

---

## 📝 Notes

- **bitsandbytes** is the easiest quantization method (native in transformers)
- GPTQ/AWQ require pre-quantized checkpoints (not yet available for Qwen-2.5-VL)
- For faster testing, use `--max_samples 100` before full runs
- All results auto-save with unique filenames (won't overwrite)

---

## ❓ Troubleshooting

**Out of memory with 4-bit:**
```bash
# Reduce batch size or use gradient checkpointing
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

**Quantization fails:**
```bash
# Install/update bitsandbytes
pip install --upgrade bitsandbytes
```

**Want to test on subset first:**
```bash
python evaluate.py --model_size 7B --quantization bitsandbytes --bits 4 --max_samples 100
```
