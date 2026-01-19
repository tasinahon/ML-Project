# Important: Gemma Model Selection

## Available Gemma Models

The model **`google/gemma-3-4b-it` does NOT exist**. 

Available Gemma instruct models are:

| Model | Size | Speed | Quality | Memory |
|-------|------|-------|---------|--------|
| `google/gemma-2-2b-it` | 2B | ⚡⚡⚡ Fastest | ⭐⭐ Basic | ~4GB |
| `google/gemma-2-9b-it` | 9B | ⚡⚡ Fast | ⭐⭐⭐⭐ Good | ~18GB |
| `google/gemma-2-27b-it` | 27B | ⚡ Slow | ⭐⭐⭐⭐⭐ Best | ~54GB |

## Current Configuration

The scripts are now using **`google/gemma-2-2b-it`** (2B model) as it's:
- ✅ Closest to the requested ~4B size
- ✅ Fast inference (~2-3x faster than 9B)
- ✅ Low memory footprint
- ⚠️ Less accurate than 9B for complex judgments

## Recommendation

For **best accuracy**, I recommend using **`google/gemma-2-9b-it`** instead:

```bash
# Edit test_llm_judge.sh or run_llm_judge_full.sh
# Change this line:
JUDGE_MODEL="google/gemma-2-2b-it"

# To this:
JUDGE_MODEL="google/gemma-2-9b-it"
```

Or run manually:
```bash
python llm_judge_eval.py \
    --response_file results/responses_qwen25vl_7b_fp16.json \
    --max_samples 10 \
    --judge_model google/gemma-2-9b-it \
    --device auto
```

## Trade-offs

### gemma-2-2b-it (Current):
- ✅ Fast (10-20 samples/min)
- ✅ Low memory (~4GB)
- ⚠️ May miss subtle safety violations
- ⚠️ Less context understanding

### gemma-2-9b-it (Recommended):
- ✅ Better accuracy
- ✅ Understands complex contexts
- ✅ More reliable judgments
- ⚠️ Slower (5-10 samples/min)
- ⚠️ Higher memory (~18GB)

### gemma-2-27b-it (Best quality):
- ✅ Best accuracy
- ✅ Most reliable
- ⚠️ Very slow (2-3 samples/min)
- ⚠️ High memory (~54GB)

## What to Do

1. **Try 2B first** - Run the test to see if quality is acceptable
2. **If results look good** - Continue with 2B for speed
3. **If judgments seem off** - Switch to 9B for better quality

You can always compare results from both models later!
