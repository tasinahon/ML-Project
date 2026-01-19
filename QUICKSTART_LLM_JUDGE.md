# Quick Start: LLM-as-Judge Evaluation

## ✅ Implementation Complete!

I've created a comprehensive LLM-as-Judge evaluation system using Gemma-2-9b-it to properly evaluate your VLM responses.

## 📁 Files Created

1. **`llm_judge_eval.py`** - Main evaluation script
2. **`test_llm_judge.sh`** - Quick test (10 samples)
3. **`run_llm_judge_full.sh`** - Full evaluation (all models)
4. **`docs/LLM_JUDGE_GUIDE.md`** - Comprehensive documentation

## 🚀 How to Use

### Step 1: Test with 10 Samples

```bash
./test_llm_judge.sh
```

This will:
- Activate your venv
- Test on 10 samples from FP16 model
- Save results to `results/llm_judge_test/`
- Show comparison with string matching

**Expected time:** 5-10 minutes

### Step 2: Review Results

Check the output files:
```bash
ls -lh results/llm_judge_test/
```

You'll see:
- `llm_judge_responses_qwen25vl_7b_fp16_*.json` - Detailed judgments
- `llm_judge_metrics_responses_qwen25vl_7b_fp16_*.json` - Summary metrics

### Step 3: Run Full Evaluation

Once satisfied with test results:

```bash
./run_llm_judge_full.sh
```

This evaluates all models:
- ✅ Qwen2.5-VL 7B FP16
- ✅ Qwen2.5-VL 7B 4-bit Quantized
- ✅ Qwen2.5-VL 7B WANDA 50% (if exists)
- ✅ Qwen2.5-VL 3B (if exists)

**Expected time:** 3-5 hours for full dataset

## 🎯 Key Features

### 1. Multimodal Evaluation
- Analyzes **image + query + response** together
- Uses actual safety labels from HoliSafe dataset

### 2. HoliSafe-Aligned Prompts
- Uses exact evaluation prompt from the paper
- Follows research methodology

### 3. Comprehensive Metrics
- **Attack Success Rate (ASR)** - % of unsafe queries model answered
- **Refusal Rate** - % of queries where model refused
- **Category-wise breakdown**
- **Comparison with string matching**

### 4. Safety Label Mapping
```
Unsafe inputs: UI+UT, UI+ST, SI+UT, SI+ST→U
Safe inputs: SI+ST→S
```

## 📊 Expected Output

```
==========================================
EVALUATION SUMMARY
==========================================
Total samples evaluated: 10

LLM Judge Results:
  Safe responses: 7
  Unsafe responses: 3
  Refusal Rate: 70.00%
  Attack Success Rate: 30.00%

String Matching Results:
  Refusal count: 5
  Refusal Rate: 50.00%
  Attack Success Rate: 50.00%

Method Comparison:
  Agreement Rate: 80.00%
  Disagreements: 2
==========================================
```

## 🔧 Manual Usage

For custom evaluation:

```bash
source venv/bin/activate

python llm_judge_eval.py \
    --response_file results/responses_qwen25vl_7b_fp16.json \
    --max_samples 50 \
    --judge_model google/gemma-2-9b-it \
    --output_dir results/my_test \
    --device cuda
```

## 💡 Why This is Better

### String Matching Problems:
- ❌ Only checks for refusal keywords
- ❌ Misses context (e.g., "I'm sorry to hear that" vs "I'm sorry I cannot help")
- ❌ Can't understand image content
- ❌ High false positive rate

### LLM Judge Advantages:
- ✅ Understands full context
- ✅ Analyzes image + text together
- ✅ Detects subtle jailbreaks
- ✅ Aligned with research methodology
- ✅ More accurate ASR calculation

## 📈 Next Steps

1. **Run test** → Verify it works
2. **Check 10 sample judgments** → Validate quality
3. **Run full evaluation** → Get complete metrics
4. **Compare models** → FP16 vs Quantized vs Compressed
5. **Analyze differences** → Where does quantization affect safety?

## ⚙️ Configuration Options

### Different Judge Models
```bash
# Smaller, faster (current)
--judge_model google/gemma-2-9b-it

# Larger, more accurate
--judge_model google/gemma-2-27b-it
```

### Sample Sizes
```bash
# Quick test
--max_samples 10

# Medium validation
--max_samples 100

# Full evaluation
# (omit --max_samples)
```

### Device Selection
```bash
--device auto    # Automatic (recommended)
--device cuda    # Force GPU
--device cpu     # Force CPU (slow)
```

## 📖 Full Documentation

See [docs/LLM_JUDGE_GUIDE.md](docs/LLM_JUDGE_GUIDE.md) for:
- Detailed explanation of evaluation logic
- Troubleshooting guide
- Performance optimization tips
- Alternative judge models

## 🎉 Ready to Start!

Just run:
```bash
./test_llm_judge.sh
```

And you're good to go! 🚀
