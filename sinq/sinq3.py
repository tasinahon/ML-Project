"""
sinq3.py – Runnability & Effectiveness Test of SINQ on Qwen2.5-VL-3B-Instruct

This script tests whether SINQ (Sinkhorn-Normalized Quantization) can be
applied to Qwen/Qwen2.5-VL-3B-Instruct, a full-precision (BF16) 3B
Vision-Language model from Qwen.

This probes a clean quantization scenario:
  - The base model is an unquantized BF16 VL model (3B parameters).
  - SINQ is applied directly to compress all nn.Linear layers to 4-bit.
  - No safety head; we focus on VL generation quality and memory savings.

Test plan:
  1. Whether SINQ can quantize the model's linear layers without errors.
  2. Whether the quantized model can still generate coherent text for
     image-understanding tasks (two prompts).
  3. Memory footprint comparison before/after SINQ.

Usage:
    python sinq3.py
"""

import os
from dotenv import load_dotenv
import gc
import sys
import time
import traceback
from contextlib import contextmanager

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ── SINQ imports ──────────────────────────────────────────────────────────────
from sinq.patch_model import AutoSINQHFModel
from sinq.sinqlinear import BaseQuantizeConfig, SINQLinear

load_dotenv()  # Load environment variables from .env file if present
MY_CACHE_PATH = os.getenv("MY_CACHE_PATH", "./model_cache")  # Default cache

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Public demo image from the Qwen model card
TEST_IMAGE_URL = (
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
)
TEST_PROMPT = "Describe this image."

# A second prompt to test a different capability
TEST_PROMPT_2 = "What colors are prominent in this image?"

# ── Helpers ───────────────────────────────────────────────────────────────────

def separator(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


def gpu_mem_mb() -> tuple[float, float]:
    """Return (allocated, reserved) GPU memory in MB."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (
        torch.cuda.memory_allocated() / 1e6,
        torch.cuda.memory_reserved() / 1e6,
    )


def report_memory(label: str) -> None:
    alloc, resv = gpu_mem_mb()
    print(f"  [{label}] GPU memory – allocated: {alloc:,.0f} MB, reserved: {resv:,.0f} MB")


@contextmanager
def timer_ctx(label: str):
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    print(f"  ⏱  {label}: {elapsed:.2f}s")


def count_linear_layers(model) -> dict[str, int]:
    """Count nn.Linear vs SINQLinear layers in the model."""
    counts = {"nn.Linear": 0, "SINQLinear": 0}
    for _, m in model.named_modules():
        if isinstance(m, SINQLinear):
            counts["SINQLinear"] += 1
        elif isinstance(m, torch.nn.Linear):
            counts["nn.Linear"] += 1
    return counts


def prepare_inputs(processor, prompt: str = TEST_PROMPT):
    """Build model inputs from the test image + prompt."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": TEST_IMAGE_URL},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs


def run_generation(model, inputs, processor, max_new_tokens: int = 128) -> str | None:
    """Generate text from the model and return decoded string."""
    try:
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        texts = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return texts[0]
    except Exception as e:
        print(f"  ⚠  Generation failed: {e}")
        traceback.print_exc()
        return None


# ── Main Test Flow ────────────────────────────────────────────────────────────

def main():
    results = {
        "load_model": False,
        "baseline_generation_1": False,
        "baseline_generation_2": False,
        "sinq_quantize": False,
        "quantized_generation_1": False,
        "quantized_generation_2": False,
    }

    # ── 0. Environment check ──────────────────────────────────────────────
    separator("0. Environment Check")
    print(f"  Python       : {sys.version}")
    print(f"  PyTorch      : {torch.__version__}")
    print(f"  CUDA avail   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device  : {torch.cuda.get_device_name(0)}")
        print(f"  CUDA memory  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    report_memory("initial")

    # ── 1. Load model & processor ─────────────────────────────────────────
    separator("1. Load Qwen2.5-VL-3B-Instruct (BF16 baseline)")
    print(f"  Model: {MODEL_ID}")
    print("  Note: This is the full-precision (BF16) model – no prior quantization.")
    with timer_ctx("model loading"):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype="auto",
            device_map="auto",
            cache_dir=MY_CACHE_PATH
        )
        processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=MY_CACHE_PATH)
    results["load_model"] = True
    report_memory("after model load")

    layer_counts_before = count_linear_layers(model)
    print(f"  Linear layer counts: {layer_counts_before}")

    # ── 2. Prepare inputs ─────────────────────────────────────────────────
    separator("2. Prepare Inputs")
    with timer_ctx("input preparation"):
        inputs_1 = prepare_inputs(processor, TEST_PROMPT).to(model.device)
        inputs_2 = prepare_inputs(processor, TEST_PROMPT_2).to(model.device)
    print(f"  Prompt 1        : {TEST_PROMPT!r}")
    print(f"  input_ids shape : {inputs_1.input_ids.shape}")
    print(f"  Prompt 2        : {TEST_PROMPT_2!r}")
    print(f"  input_ids shape : {inputs_2.input_ids.shape}")

    # ── 3. Baseline inference (BF16) ──────────────────────────────────────
    separator("3. Baseline – Text Generation (BF16, pre-SINQ)")

    with timer_ctx("baseline generation – prompt 1"):
        baseline_text_1 = run_generation(model, inputs_1, processor)
    if baseline_text_1 is not None:
        results["baseline_generation_1"] = True
        print(f"  Generated text (prompt 1):\n    {baseline_text_1[:500]}")
    else:
        print("  ✗ Baseline generation (prompt 1) did not succeed.")

    with timer_ctx("baseline generation – prompt 2"):
        baseline_text_2 = run_generation(model, inputs_2, processor)
    if baseline_text_2 is not None:
        results["baseline_generation_2"] = True
        print(f"  Generated text (prompt 2):\n    {baseline_text_2[:500]}")
    else:
        print("  ✗ Baseline generation (prompt 2) did not succeed.")

    baseline_mem = gpu_mem_mb()
    report_memory("after baseline inference")

    # ── 4. SINQ Quantization ──────────────────────────────────────────────
    separator("4. Apply SINQ 4-bit Quantization")
    print("  Quantization config: nbits=4, group_size=64, tiling_mode=1D, method=sinq")
    print("  SINQ will quantize all nn.Linear layers (except lm_head) to 4-bit.")

    # Clear caches before quantization
    torch.cuda.empty_cache()
    gc.collect()

    quant_config = BaseQuantizeConfig(
        nbits=4,
        group_size=64,
        tiling_mode="1D",
        method="sinq",
    )

    try:
        with timer_ctx("SINQ quantization"):
            AutoSINQHFModel.quantize_model(
                model,
                tokenizer=processor.tokenizer if hasattr(processor, "tokenizer") else processor,
                quant_config=quant_config,
                compute_dtype=torch.float16,
                device=DEVICE,
            )
        results["sinq_quantize"] = True
        print("  ✓ SINQ quantization completed successfully!")
    except Exception as e:
        print(f"  ✗ SINQ quantization FAILED: {e}")
        traceback.print_exc()
        # Even if quantization fails, report what we can
        _print_summary(results, baseline_mem, gpu_mem_mb(),
                        layer_counts_before, count_linear_layers(model),
                        baseline_text_1, None, baseline_text_2, None)
        return

    report_memory("after quantization")

    layer_counts_after = count_linear_layers(model)
    print(f"  Linear layer counts after quantization: {layer_counts_after}")

    # ── 5. Quantized inference ────────────────────────────────────────────
    separator("5. Quantized – Text Generation (SINQ 4-bit)")

    # Re-prepare inputs in case device changed
    inputs_1 = prepare_inputs(processor, TEST_PROMPT).to(model.device)
    inputs_2 = prepare_inputs(processor, TEST_PROMPT_2).to(model.device)

    with timer_ctx("quantized generation – prompt 1"):
        quant_text_1 = run_generation(model, inputs_1, processor)
    if quant_text_1 is not None:
        results["quantized_generation_1"] = True
        print(f"  Generated text (prompt 1):\n    {quant_text_1[:500]}")
    else:
        print("  ✗ Quantized generation (prompt 1) did not succeed.")

    with timer_ctx("quantized generation – prompt 2"):
        quant_text_2 = run_generation(model, inputs_2, processor)
    if quant_text_2 is not None:
        results["quantized_generation_2"] = True
        print(f"  Generated text (prompt 2):\n    {quant_text_2[:500]}")
    else:
        print("  ✗ Quantized generation (prompt 2) did not succeed.")

    quant_mem = gpu_mem_mb()
    report_memory("after quantized inference")

    # ── 6. Effectiveness comparison ───────────────────────────────────────
    _print_summary(results, baseline_mem, quant_mem,
                    layer_counts_before, layer_counts_after,
                    baseline_text_1, quant_text_1,
                    baseline_text_2, quant_text_2)


def _compute_jaccard(text_a: str, text_b: str) -> float:
    """Word-level Jaccard similarity between two strings."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    union = words_a | words_b
    if not union:
        return 0.0
    return len(words_a & words_b) / len(union)


def _print_summary(results, baseline_mem, quant_mem,
                    counts_before, counts_after,
                    baseline_text_1, quant_text_1,
                    baseline_text_2, quant_text_2):
    separator("6. Effectiveness Comparison")

    # 6a. Generation quality (prompt 1)
    if baseline_text_1 and quant_text_1:
        jaccard_1 = _compute_jaccard(baseline_text_1, quant_text_1)
        print(f"  Prompt 1 Jaccard similarity : {jaccard_1:.4f}")
        print(f"  Prompt 1 baseline length    : {len(baseline_text_1)} chars")
        print(f"  Prompt 1 quantized length   : {len(quant_text_1)} chars")
    elif baseline_text_1:
        print("  Prompt 1: baseline succeeded but quantized failed.")
    else:
        print("  Prompt 1: baseline did not produce output.")

    # 6b. Generation quality (prompt 2)
    if baseline_text_2 and quant_text_2:
        jaccard_2 = _compute_jaccard(baseline_text_2, quant_text_2)
        print(f"  Prompt 2 Jaccard similarity : {jaccard_2:.4f}")
        print(f"  Prompt 2 baseline length    : {len(baseline_text_2)} chars")
        print(f"  Prompt 2 quantized length   : {len(quant_text_2)} chars")
    elif baseline_text_2:
        print("  Prompt 2: baseline succeeded but quantized failed.")
    else:
        print("  Prompt 2: baseline did not produce output.")

    # 6c. Coherence check – does the quantized output look like real language?
    if quant_text_1:
        words = quant_text_1.split()
        unique_ratio = len(set(words)) / max(len(words), 1)
        print(f"  Prompt 1 unique-word ratio  : {unique_ratio:.3f}  (>0.3 = likely coherent)")
    if quant_text_2:
        words = quant_text_2.split()
        unique_ratio = len(set(words)) / max(len(words), 1)
        print(f"  Prompt 2 unique-word ratio  : {unique_ratio:.3f}  (>0.3 = likely coherent)")

    # ── 7. Summary ────────────────────────────────────────────────────────
    separator("7. Summary")
    print("  Test Results:")
    status_sym = {True: "✓ PASS", False: "✗ FAIL"}
    for test_name, passed in results.items():
        print(f"    {test_name:30s} : {status_sym[passed]}")

    print(f"\n  Model                     : {MODEL_ID}")
    print(f"  Layer Counts Before Quant : {counts_before}")
    print(f"  Layer Counts After Quant  : {counts_after}")
    print(f"  Baseline GPU alloc        : {baseline_mem[0]:,.0f} MB")
    print(f"  Quantized GPU alloc       : {quant_mem[0]:,.0f} MB")
    if baseline_mem[0] > 0:
        ratio = quant_mem[0] / baseline_mem[0]
        saving_pct = (1 - ratio) * 100
        print(f"  Memory ratio (quant/base) : {ratio:.3f}  ({saving_pct:+.1f}% saving)")

    all_passed = all(results.values())
    print(f"\n  {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print(f"  SINQ runnability on {MODEL_ID}: {'CONFIRMED' if results['sinq_quantize'] else 'FAILED'}")
    if results["quantized_generation_1"] or results["quantized_generation_2"]:
        print("  SINQ effectiveness: Quantized model retains generation capabilities.")
    elif results["sinq_quantize"]:
        print("  SINQ effectiveness: Quantization succeeded but inference tests failed.")
    else:
        print("  SINQ effectiveness: Could not be evaluated (quantization failed).")


if __name__ == "__main__":
    main()
