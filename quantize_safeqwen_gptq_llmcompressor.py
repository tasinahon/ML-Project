"""
Quantize SafeQwen2.5-VL-7B using llm-compressor GPTQ
GPTQ is more stable than AWQ for VLMs - uses weight-only quantization
"""
import argparse
import base64
from io import BytesIO
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot

def main():
    parser = argparse.ArgumentParser(description='Quantize SafeQwen with GPTQ using llm-compressor')
    parser.add_argument('--model_id', type=str, default='etri-vilab/SafeQwen2.5-VL-7B',
                        help='Model ID or path')
    parser.add_argument('--output_dir', type=str, default='safeqwen_gptq_w4a16',
                        help='Output directory for quantized model')
    parser.add_argument('--num_samples', type=int, default=256,
                        help='Number of calibration samples')
    parser.add_argument('--max_seq_length', type=int, default=2048,
                        help='Maximum sequence length')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    print(f"Loading model: {args.model_id}")
    
    # Load model in FP16
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True
    )
    
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    print(f"Model architecture: {model.__class__.__name__}")
    
    # Load calibration dataset (HoliSafe-Bench for safety relevance)
    print(f"\nLoading calibration dataset...")
    dataset = load_dataset(
        "etri-vilab/holisafe-bench",
        split='test'
    )
    
    # Take subset for calibration
    if len(dataset) > args.num_samples:
        dataset = dataset.select(range(args.num_samples))
    
    print(f"Using {len(dataset)} samples for calibration")
    
    # Preprocess and tokenize the dataset
    def preprocess_and_tokenize(example):
        """Tokenize multimodal (image + text) prompts for calibration"""
        # Convert PIL image to base64
        buffered = BytesIO()
        example['image'].save(buffered, format="PNG")
        encoded_image = base64.b64encode(buffered.getvalue())
        encoded_image_text = encoded_image.decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image_text}"
        
        # Create conversation format with image and text
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": base64_qwen},
                    {"type": "text", "text": example['query']},
                ],
            }
        ]
        
        # Apply chat template
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Tokenize with image (no truncation to preserve all image tokens)
        return processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            truncation=False,  # Disable truncation to avoid cutting image tokens
        )
    
    print("Preprocessing dataset (converting images to base64 and tokenizing)...")
    dataset = dataset.map(preprocess_and_tokenize, remove_columns=dataset.column_names)
    
    # Define data collator for batching
    def data_collator(batch):
        """Collate batch for calibration"""
        assert len(batch) == 1  # Calibration uses batch size 1
        return {key: torch.tensor(value) for key, value in batch[0].items()}
    
    # Create GPTQ recipe
    # Targets all Linear layers except vision encoder and safety head
    print("\nCreating GPTQ quantization recipe...")
    recipe = GPTQModifier(
        targets="Linear",
        scheme="W4A16",  # 4-bit weights, 16-bit activations
        ignore=[
            "lm_head",           # Keep output head in FP16
            "re:visual.*",       # Exclude vision encoder
            "re:model.visual.*", # Alternative vision encoder path
            "img_safety_head.*"  # Preserve SafeQwen safety head
        ],
    )
    
    print(f"Recipe: GPTQ W4A16")
    print(f"Targets: All Linear layers")
    print(f"Ignored: lm_head, visual encoder, img_safety_head")
    print(f"Algorithm: Weight-only quantization (more stable than AWQ for VLMs)")
    
    # Apply GPTQ quantization with oneshot
    print("\nStarting GPTQ quantization...")
    print("This will take 30-45 minutes depending on calibration samples")
    
    oneshot(
        model=model,
        tokenizer=args.model_id,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=len(dataset),
        trust_remote_code_model=True,
        data_collator=data_collator,
        # Sequential processing for memory efficiency
        sequential_targets=["Qwen2_5_VLDecoderLayer"],
    )
    
    print("\nQuantization completed!")
    
    # Save quantized model
    print(f"\nSaving quantized model to {args.output_dir}...")
    model.save_pretrained(
        args.output_dir,
        save_compressed=True  # Save in compressed-tensors format
    )
    processor.save_pretrained(args.output_dir)
    
    print(f"✓ Model saved to {args.output_dir}")
    print(f"✓ Ready for vLLM inference or evaluation")
    
    # Test generation - load model fresh for inference
    print("\nTesting generation on quantized model...")
    print("Loading quantized model for inference...")
    
    # Reload model with proper device placement for inference
    del model  # Free memory
    torch.cuda.empty_cache()
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.output_dir,
        device_map="auto",
        torch_dtype="auto",
    )
    
    test_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello, how are you?"},
            ],
        }
    ]
    
    text = processor.apply_chat_template(
        test_messages, tokenize=False, add_generation_prompt=True
    )
    
    inputs = processor(text=[text], images=None, return_tensors="pt")
    # Move inputs to same device as model's first parameter
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False
        )
    
    response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    print(f"\nGeneration test:")
    print(f"Input: Hello, how are you?")
    print(f"Output: {response[-200:]}")  # Last 200 chars
    
    print("\n" + "="*60)
    print("GPTQ quantization completed successfully!")
    print(f"Quantized model saved to: {args.output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
