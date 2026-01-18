"""
Main evaluation script for HoliSafe-Bench on Qwen-2.5-VL models
Measures Attack Success Rate, Refusal Rate, and optionally Refusal Embeddings
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import torch

from dataset_loader import HoliSafeBenchLoader
from model_wrapper import Qwen25VLWrapper
from metrics import SafetyMetrics


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen-2.5-VL models on HoliSafe-Bench"
    )
    
    parser.add_argument(
        "--model_size",
        type=str,
        choices=["3B", "7B", "both"],
        default="3B",
        help="Model size to evaluate (3B, 7B, or both)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save evaluation results"
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for testing)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (currently only 1 supported)"
    )
    
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    
    parser.add_argument(
        "--use_llm_judge",
        action="store_true",
        help="Use LLM as judge for refusal detection"
    )
    
    parser.add_argument(
        "--judge_model",
        type=str,
        choices=["gpt-4", "claude-3.5-sonnet"],
        default=None,
        help="LLM model to use as judge"
    )
    
    parser.add_argument(
        "--save_responses",
        action="store_true",
        default=True,
        help="Save individual responses to file"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run model on"
    )
    
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        default=False,
        help="Use flash attention for faster inference"
    )
    
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["bitsandbytes", "gptq", "awq", None],
        default=None,
        help="Quantization method (None for FP16 baseline)"
    )
    
    parser.add_argument(
        "--bits",
        type=int,
        choices=[4, 8],
        default=4,
        help="Quantization bits (4 or 8)"
    )
    
    return parser.parse_args()


def evaluate_model(
    model_size: str,
    dataset_loader: HoliSafeBenchLoader,
    args,
    output_dir: Path
):
    """
    Evaluate a single model on the dataset
    
    Args:
        model_size: Size of the model ("3B" or "7B")
        dataset_loader: Loaded dataset
        args: Command line arguments
        output_dir: Directory to save results
    """
    print(f"\n{'='*80}")
    print(f"Evaluating Qwen-2.5-VL-{model_size}")
    print(f"{'='*80}\n")
    
    # Initialize model
    model = Qwen25VLWrapper(
        model_size=model_size,
        device=args.device,
        use_flash_attention=args.use_flash_attention,
        quantization=args.quantization,
        bits=args.bits
    )
    
    # Prepare samples
    num_samples = args.max_samples if args.max_samples else len(dataset_loader)
    samples = [dataset_loader.get_sample(i) for i in range(num_samples)]
    
    print(f"Evaluating on {num_samples} samples...")
    
    # Generate responses
    results = model.batch_generate(
        samples=samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        show_progress=True
    )
    
    # Save raw responses if requested
    if args.save_responses:
        quant_suffix = f"_{args.quantization}{args.bits}bit" if args.quantization else "_fp16"
        responses_file = output_dir / f"responses_qwen25vl_{model_size.lower()}{quant_suffix}.json"
        with open(responses_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved responses to {responses_file}")
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics_calculator = SafetyMetrics(
        use_llm_judge=args.use_llm_judge,
        judge_model=args.judge_model
    )
    
    metrics = metrics_calculator.compute_metrics(
        results,
        use_llm=args.use_llm_judge
    )
    
    # Save metrics
    quant_suffix = f"_{args.quantization}{args.bits}bit" if args.quantization else "_fp16"
    metrics_file = output_dir / f"metrics_qwen25vl_{model_size.lower()}{quant_suffix}.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Saved metrics to {metrics_file}")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Results for Qwen-2.5-VL-{model_size}")
    print(f"{'='*80}")
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"Refusal Rate: {metrics['refusal_rate']:.2f}%")
    print(f"Attack Success Rate: {metrics['attack_success_rate']:.2f}%")
    print(f"\nCategory-wise Metrics:")
    for category, cat_metrics in metrics['category_metrics'].items():
        print(f"  {category}:")
        print(f"    Total: {cat_metrics['total']}")
        print(f"    Refusal Rate: {cat_metrics['refusal_rate']:.2f}%")
        print(f"    ASR: {cat_metrics['asr']:.2f}%")
    
    print(f"\nSafeness Combination Metrics:")
    for safeness, safe_metrics in metrics['safeness_metrics'].items():
        print(f"  {safeness}:")
        print(f"    Total: {safe_metrics['total']}")
        print(f"    Refusal Rate: {safe_metrics['refusal_rate']:.2f}%")
        print(f"    ASR: {safe_metrics['asr']:.2f}%")
    
    # Clean up
    model.clear_cache()
    del model
    torch.cuda.empty_cache()
    
    return metrics


def main():
    """Main evaluation function"""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    config['timestamp'] = datetime.now().isoformat()
    config_file = output_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("="*80)
    print("HoliSafe-Bench Evaluation on Qwen-2.5-VL")
    print("="*80)
    print(f"Configuration: {config}")
    print("="*80)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset_loader = HoliSafeBenchLoader(use_hf_api=True)
    dataset_loader.load()
    
    # Print dataset statistics
    print("\n" + "="*80)
    print("Dataset Statistics")
    print("="*80)
    print(f"Total samples: {len(dataset_loader)}")
    # Skip detailed stats for speed - just show total
    # stats = dataset_loader.get_statistics()
    # print(f"Total samples: {stats['total_samples']}")
    # print(f"\nCategories:")
    # for cat, count in stats['categories'].items():
    #     print(f"  {cat}: {count}")
    # print(f"\nSafeness Combinations:")
    # for safe, count in stats['safeness_combinations'].items():
    #     print(f"  {safe}: {count}")
    
    # Evaluate models
    all_results = {}
    
    if args.model_size == "both":
        model_sizes = ["3B", "7B"]
    else:
        model_sizes = [args.model_size]
    
    for model_size in model_sizes:
        try:
            metrics = evaluate_model(
                model_size=model_size,
                dataset_loader=dataset_loader,
                args=args,
                output_dir=output_dir
            )
            all_results[f"qwen25vl_{model_size.lower()}"] = metrics
        except Exception as e:
            print(f"\nError evaluating {model_size} model: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save comparison if multiple models evaluated
    if len(all_results) > 1:
        comparison_file = output_dir / "comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n{'='*80}")
        print("Model Comparison")
        print(f"{'='*80}")
        for model_name, metrics in all_results.items():
            print(f"\n{model_name}:")
            print(f"  Refusal Rate: {metrics['refusal_rate']:.2f}%")
            print(f"  Attack Success Rate: {metrics['attack_success_rate']:.2f}%")
    
    print(f"\n{'='*80}")
    print(f"Evaluation complete! Results saved to {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
