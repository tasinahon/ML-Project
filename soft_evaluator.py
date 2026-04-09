"""
Soft Evaluator — Runs GemmaJudge over response JSON files and writes
classifications alongside the original data.

Usage:
    python soft_evaluator.py <response_file> [<response_file> ...]

Examples:
    python soft_evaluator.py llama32_results/responses_llama32_11b_vision_bitsandbytes4bit_fp4.json
    python soft_evaluator.py safeqwen_results/responses_safeqwen_7b_fp16.json results/responses_qwen25vl_7b_fp16.json
    python soft_evaluator.py llama32_results/*.json safeqwen_results/*.json results/responses_*.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

from gemma_softjudge import GemmaJudge


def build_output_path(input_path: str) -> str:
    """
    Given an input path like `dir/responses_foo.json`,
    return `dir/responses_foo_softjudge.json`.
    """
    p = Path(input_path)
    return str(p.with_name(p.stem + "_softjudge" + p.suffix))


def load_responses(path: str) -> list:
    """Load a response JSON file (expected to be a JSON array)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}")
    return data


def evaluate_file(judge: GemmaJudge, input_path: str, batch_size: int) -> None:
    """
    Run the soft judge on a single response file, write results to
    <original_name>_softjudge.json in the same directory.
    """
    output_path = build_output_path(input_path)
    print(f"\n{'='*60}")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}")

    data = load_responses(input_path)
    print(f"  Loaded {len(data)} entries")

    # Prepare samples for batch_classify (needs 'query' and 'response' keys)
    samples = []
    for entry in data:
        samples.append({
            "query": entry.get("query", ""),
            "response": entry.get("response", ""),
        })

    # Run classification
    classifications = judge.batch_classify(
        samples,
        show_progress=True,
        batch_size=batch_size,
    )

    # Merge classification into original data
    results = []
    for entry, classification in zip(data, classifications):
        result = dict(entry)  # preserve all original fields
        result["softjudge_classification"] = classification
        results.append(result)

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    from collections import Counter
    counts = Counter(classifications)
    print(f"\n  Summary for {os.path.basename(input_path)}:")
    for label, count in sorted(counts.items()):
        pct = 100.0 * count / len(classifications)
        print(f"    {label:25s}: {count:5d}  ({pct:.1f}%)")
    print(f"  Results written to {output_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run GemmaJudge soft evaluation on response JSON files."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="One or more response JSON files to evaluate "
             "(e.g. llama32_results/responses_*.json)",
    )
    parser.add_argument(
        "--model",
        default="google/gemma-2-2b-it",
        help="Gemma model to use for judging (default: google/gemma-2-2b-it)",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable 8-bit quantization for the judge model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of samples to evaluate per batch (default: 8)",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        print("Error: --batch-size must be >= 1", file=sys.stderr)
        sys.exit(1)

    # Validate all files exist before loading the model
    for path in args.files:
        if not os.path.isfile(path):
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(1)

    # Initialise judge once and reuse across all files
    judge = GemmaJudge(
        model_name=args.model,
        quantization=not args.no_quantize,
    )

    for path in args.files:
        try:
            evaluate_file(judge, path, batch_size=args.batch_size)
        except Exception as e:
            print(f"Error processing {path}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    judge.clear_cache()
    print("Done — all files processed.")


if __name__ == "__main__":
    main()
