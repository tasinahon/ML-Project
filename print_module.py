from typing import List

import argparse
import torch
from torch import nn
from transformers import AutoModelForVision2Seq, BitsAndBytesConfig


def get_names_of_modules(model: nn.Module) -> List[str]:
    """
    Retrieve a list of module names from a given model.

    Args:
        model: The PyTorch model from which to retrieve module names.

    Returns:
        A list of module names as strings.

    Example:
        >>> model = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
        >>> names = get_names_of_modules(model)
        >>> print(names)
        ['', '0', '1']
    """
    return [name for name, _ in model.named_modules()]


def load_safeqwen_model(use_fp16: bool, quant_type: str) -> nn.Module:
    model_id = "etri-vilab/SafeQwen2.5-VL-7B"

    if use_fp16:
        return AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    return AutoModelForVision2Seq.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print SafeQwen module names")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Load FP16 instead of 4-bit bitsandbytes",
    )
    parser.add_argument(
        "--quant_type",
        type=str,
        default="nf4",
        choices=["nf4", "fp4"],
        help="4-bit quantization type (used when --fp16 is not set)",
    )
    args = parser.parse_args()

    model = load_safeqwen_model(use_fp16=args.fp16, quant_type=args.quant_type)
    module_names = get_names_of_modules(model)

    print("Module Names in SafeQwen:")
    for item in module_names:
        print(item)
