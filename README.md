# HoliSafe-Bench Evaluation on Qwen-2.5-VL

This project evaluates Qwen-2.5-VL models (3B and 7B) on the HoliSafe-Bench dataset to measure:
- **Attack Success Rate (ASR)**: Percentage of unsafe queries that the model answered instead of refusing
- **Refusal Rate**: Percentage of queries where the model refused to answer
- **Refusal Embeddings** (optional): Embeddings of refusal responses for analysis

## Dataset

HoliSafe-Bench is a holistic safety benchmark for Vision-Language Models covering all five safeness combinations:
1. **UI+UT**: Unsafe Image + Unsafe Text
2. **UI+ST**: Unsafe Image + Safe Text
3. **SI+UT**: Safe Image + Unsafe Text
4. **SI+ST→U**: Safe Image + Safe Text → Unsafe outcome
5. **SI+ST→S**: Safe Image + Safe Text → Safe outcome

Dataset source: [etri-vilab/holisafe-bench](https://huggingface.co/datasets/etri-vilab/holisafe-bench)

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for 7B model)
- 16GB+ RAM for 3B model, 32GB+ for 7B model

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# For flash attention support (optional, faster inference)
pip install flash-attn --no-build-isolation
```

## Usage

### Quick Start - Evaluate 3B Model

```bash
python evaluate.py --model_size 3B --max_samples 100
```

### Evaluate 7B Model

```bash
python evaluate.py --model_size 7B
```

### Evaluate Both Models

```bash
python evaluate.py --model_size both
```

### Full Options

```bash
python evaluate.py \
    --model_size 3B \
    --output_dir ./results \
    --max_samples 1000 \
    --max_new_tokens 512 \
    --temperature 0.7 \
    --save_responses \
    --device auto \
    --use_flash_attention
```

### With LLM Judge (GPT-4 or Claude)

For more accurate refusal detection, use an LLM as judge:

```bash
# Set API key
export OPENAI_API_KEY=your_key_here
# or
export ANTHROPIC_API_KEY=your_key_here

# Run evaluation
python evaluate.py \
    --model_size 3B \
    --use_llm_judge \
    --judge_model gpt-4
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_size` | str | `3B` | Model size: `3B`, `7B`, or `both` |
| `--output_dir` | str | `./results` | Directory to save results |
| `--max_samples` | int | `None` | Max samples to evaluate (all if None) |
| `--max_new_tokens` | int | `512` | Max tokens to generate |
| `--temperature` | float | `0.7` | Sampling temperature |
| `--use_llm_judge` | flag | `False` | Use LLM as judge for refusal detection |
| `--judge_model` | str | `None` | LLM judge: `gpt-4` or `claude-3.5-sonnet` |
| `--save_responses` | flag | `True` | Save individual responses |
| `--device` | str | `auto` | Device: `auto`, `cuda`, or `cpu` |
| `--use_flash_attention` | flag | `True` | Use flash attention |

## Output Files

Results are saved in the `output_dir`:

```
results/
├── config.json                      # Evaluation configuration
├── metrics_qwen25vl_3b.json        # Metrics for 3B model
├── metrics_qwen25vl_7b.json        # Metrics for 7B model
├── responses_qwen25vl_3b.json      # Individual responses (if --save_responses)
├── responses_qwen25vl_7b.json      # Individual responses (if --save_responses)
└── comparison.json                  # Comparison (if evaluating both models)
```

### Metrics Structure

```json
{
  "total_samples": 4031,
  "total_refusals": 2500,
  "total_attacks_successful": 1531,
  "refusal_rate": 62.01,
  "attack_success_rate": 37.99,
  "category_metrics": {
    "Illegal Activity": {
      "total": 500,
      "refusal_rate": 75.2,
      "asr": 24.8
    },
    ...
  },
  "safeness_metrics": {
    "UI_UT": {
      "total": 800,
      "refusal_rate": 85.5,
      "asr": 14.5
    },
    ...
  }
}
```

## Visualization

Generate plots and reports from evaluation results:

```bash
python visualize.py --model_name qwen25vl_3b --results_dir ./results
```

This generates:
- Text report with detailed metrics
- Category-wise comparison plots
- Safeness combination analysis plots

## Project Structure

```
.
├── evaluate.py           # Main evaluation script
├── dataset_loader.py     # HoliSafe-Bench dataset loader
├── model_wrapper.py      # Qwen-2.5-VL model wrapper
├── metrics.py           # Metrics computation (ASR, Refusal Rate)
├── visualize.py         # Visualization and reporting
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Expected Runtime

| Model | Samples | GPU | Time (approx) |
|-------|---------|-----|---------------|
| 3B | 100 | A100 | ~5 min |
| 3B | 4031 (full) | A100 | ~3 hours |
| 7B | 100 | A100 | ~8 min |
| 7B | 4031 (full) | A100 | ~5 hours |

*Times vary based on hardware and generation parameters*

## Memory Requirements

| Model | FP16/BF16 | Flash Attention |
|-------|-----------|-----------------|
| 3B | ~8GB VRAM | ~6GB VRAM |
| 7B | ~16GB VRAM | ~14GB VRAM |

## Troubleshooting

### Out of Memory (OOM)

1. Use smaller model (3B instead of 7B)
2. Reduce `max_new_tokens`
3. Enable flash attention: `--use_flash_attention`
4. Use CPU (very slow): `--device cpu`

### Slow inference

1. Enable flash attention: `--use_flash_attention`
2. Use GPU: `--device cuda`
3. Reduce `max_new_tokens`
4. Test on smaller subset first: `--max_samples 100`

### Import errors

```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# For qwen-vl-utils
pip install qwen-vl-utils
```

## Citation

If you use this evaluation code or the HoliSafe-Bench dataset, please cite:

```bibtex
@article{holisafe2024,
  title={HoliSafe: Holistic Safety Tuning and Benchmark for Vision-Language Models},
  author={...},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is for research purposes only. Please refer to the original HoliSafe-Bench and Qwen-2.5-VL licenses for usage restrictions.
