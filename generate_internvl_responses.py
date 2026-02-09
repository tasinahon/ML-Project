"""
Generate responses using InternVL 2.5-8B model on HoliSafe-Bench dataset
Supports both BF16 and BitsAndBytes 4-bit quantization
"""

import torch
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset
from datetime import datetime
from pathlib import Path
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def process_image(image, input_size=448, max_num=12):
    """Process PIL image for InternVL"""
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class InternVLResponseGenerator:
    """Generate responses using InternVL 2.5-8B model"""
    
    def __init__(
        self,
        model_name: str = "OpenGVLab/InternVL2_5-8B",
        precision: str = "fp16",  # "fp16", "bf16"
        quantization: str = None,  # None, "4bit", "8bit"
        device: str = "auto",
        max_new_tokens: int = 2048,
        input_size: int = 448,
        max_num: int = 12
    ):
        """
        Initialize InternVL model
        
        Args:
            model_name: HuggingFace model name
            precision: "fp16" or "bf16" (only used if quantization=None)
            quantization: None (full precision), "4bit", or "8bit"
            device: Device to use
            max_new_tokens: Maximum tokens to generate
            input_size: Image input size
            max_num: Maximum number of image tiles
        """
        self.model_name = model_name
        self.precision = precision
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.input_size = input_size
        self.max_num = max_num
        
        print(f"Loading InternVL model: {model_name}")
        if quantization:
            print(f"Using {quantization} quantization")
        else:
            print(f"Using {precision.upper()} precision")
        
        # Model loading configuration
        # Choose torch_dtype based on precision
        if quantization:
            # Quantized models use bfloat16 by default
            dtype = torch.bfloat16
        elif precision == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.bfloat16
        
        model_kwargs = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        
        # Add flash attention if available
        try:
            model_kwargs["use_flash_attn"] = True
        except:
            print("Flash attention not available, using default attention")
        
        # Add quantization config
        if quantization == "4bit":
            model_kwargs["load_in_4bit"] = True
        elif quantization == "8bit":
            model_kwargs["load_in_8bit"] = True
        else:
            # BF16/FP16 - add device placement
            if device == "auto":
                model_kwargs["device_map"] = "auto"
        
        # Load model
        self.model = AutoModel.from_pretrained(model_name, **model_kwargs).eval()
        
        # For non-quantized models, move to CUDA
        if quantization is None and device != "auto":
            self.model = self.model.cuda()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False
        )
        
        print("Model loaded successfully")
    
    def generate_response(self, image, query: str) -> str:
        """
        Generate response for a single image-query pair
        
        Args:
            image: PIL Image
            query: User query
            
        Returns:
            Model response string
        """
        # Process image
        pixel_values = process_image(image, self.input_size, self.max_num)
        
        # Convert to appropriate dtype
        if self.quantization:
            pixel_values = pixel_values.to(torch.bfloat16)
        elif self.precision == "fp16":
            pixel_values = pixel_values.to(torch.float16)
        else:
            pixel_values = pixel_values.to(torch.bfloat16)
        
        # Move to GPU if not quantized
        if self.quantization is None:
            pixel_values = pixel_values.cuda()
        
        # Format question with image token
        question = f'<image>\n{query}'
        
        # Generation config
        generation_config = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=False  # Deterministic generation
        )
        
        # Generate response
        try:
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                generation_config,
                history=None,
                return_history=False
            )
            return response
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"[Error: {str(e)}]"


def generate_responses(
    model_name: str = "OpenGVLab/InternVL2_5-8B",
    precision: str = "fp16",
    quantization: str = None,
    output_dir: str = "./results",
    max_samples: int = None,
    device: str = "auto"
):
    """
    Generate responses for HoliSafe-Bench dataset
    
    Args:
        model_name: InternVL model name
        precision: "fp16" or "bf16" (only used if quantization=None)
        quantization: None (full precision), "4bit", or "8bit"
        output_dir: Directory to save results
        max_samples: Maximum samples to process (for testing)
        device: Device to use
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("Loading HoliSafe-Bench dataset...")
    dataset = load_dataset("etri-vilab/holisafe-bench")
    test_data = dataset['test']
    
    if max_samples:
        test_data = test_data.select(range(min(max_samples, len(test_data))))
        print(f"Limited to {len(test_data)} samples")
    
    # Initialize model
    generator = InternVLResponseGenerator(
        model_name=model_name,
        precision=precision,
        quantization=quantization,
        device=device
    )
    
    # Generate responses
    responses = []
    print(f"Generating responses for {len(test_data)} samples...")
    
    for sample in tqdm(test_data, desc="Generating responses"):
        image = sample['image']
        query = sample['query']
        sample_id = sample['id']
        category = sample.get('category', 'unknown')
        safeness_combination = sample.get('safeness_combination', 'Unknown')
        
        # Generate response
        response = generator.generate_response(image, query)
        
        # Store result
        result = {
            'id': sample_id,
            'query': query,
            'response': response,
            'model': model_name,
            'precision': precision if not quantization else 'quantized',
            'quantization': quantization if quantization else 'none',
            'category': category,
            'safeness_combination': safeness_combination
        }
        responses.append(result)
    
    # Save results
    model_short = model_name.split('/')[-1].replace('.', '_').lower()
    quant_suffix = f"_{quantization}" if quantization else f"_{precision}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"responses_{model_short}{quant_suffix}_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(responses, f, indent=2)
    
    print(f"\nResponses saved to: {output_file}")
    print(f"Total responses: {len(responses)}")
    
    # Save config
    config = {
        'model': model_name,
        'precision': precision if not quantization else 'quantized',
        'quantization': quantization if quantization else 'none',
        'total_samples': len(responses),
        'timestamp': timestamp,
        'device': device
    }
    
    config_file = output_dir / f"config_{model_short}{quant_suffix}_{timestamp}.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    return output_file, responses


def main():
    parser = argparse.ArgumentParser(
        description="Generate responses using InternVL 2.5-8B on HoliSafe-Bench"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='OpenGVLab/InternVL2_5-8B',
        help='InternVL model name'
    )
    parser.add_argument(
        '--precision',
        type=str,
        choices=['fp16', 'bf16'],
        default='fp16',
        help='Precision for full-precision models (fp16 or bf16)'
    )
    parser.add_argument(
        '--quantization',
        type=str,
        choices=['none', '4bit', '8bit'],
        default='none',
        help='Quantization method (none=full precision, 4bit, 8bit)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='Output directory'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum samples to process (for testing)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use'
    )
    
    args = parser.parse_args()
    
    # Convert 'none' string to None
    quantization = None if args.quantization == 'none' else args.quantization
    
    generate_responses(
        model_name=args.model,
        precision=args.precision,
        quantization=quantization,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        device=args.device
    )


if __name__ == "__main__":
    main()
