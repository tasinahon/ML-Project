"""
Model wrapper for Qwen-2.5-VL models
Handles inference for both 3B and 7B variants
"""

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info
from typing import Dict, List, Optional
from PIL import Image
import gc


class Qwen25VLWrapper:
    """Wrapper for Qwen-2.5-VL models with quantization support"""
    
    def __init__(
        self, 
        model_size: str = "3B", 
        device: str = "auto", 
        use_flash_attention: bool = True,
        quantization: Optional[str] = None,
        bits: int = 4
    ):
        """
        Initialize the Qwen-2.5-VL model
        
        Args:
            model_size: Model size, either "3B" or "7B"
            device: Device to run the model on ("cuda", "cpu", or "auto")
            use_flash_attention: Whether to use flash attention for faster inference
            quantization: Quantization method ("bitsandbytes", "gptq", "awq", or None for FP16)
            bits: Quantization bits (4 or 8)
        """
        assert model_size in ["3B", "7B"], f"Model size must be '3B' or '7B', got {model_size}"
        
        self.model_size = model_size
        self.model_name = f"Qwen/Qwen2.5-VL-{model_size}-Instruct"
        self.quantization = quantization
        self.bits = bits
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        precision_str = f"{quantization}-{bits}bit" if quantization else "FP16"
        print(f"Loading {self.model_name} on {self.device} ({precision_str})...")
        
        # Load model with quantization
        if quantization == "bitsandbytes":
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=(bits == 4),
                load_in_8bit=(bits == 8),
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        
        elif quantization == "gptq":
            # GPTQ quantized model (requires pre-quantized checkpoint)
            # For now, we'll use auto-gptq if available
            try:
                from auto_gptq import AutoGPTQForCausalLM
                print("Warning: GPTQ for VLMs may require custom implementation")
                # Fall back to bitsandbytes for now
                print("Falling back to bitsandbytes...")
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(load_in_4bit=True)
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            except ImportError:
                print("auto-gptq not installed. Falling back to bitsandbytes...")
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(load_in_4bit=True)
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True
                )
        
        elif quantization == "awq":
            # AWQ quantized model (requires pre-quantized checkpoint)
            print("Warning: AWQ for VLMs may require custom implementation")
            print("Falling back to bitsandbytes...")
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(load_in_4bit=True)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        
        else:
            # No quantization (FP16/BF16)
            if use_flash_attention and self.device == "cuda":
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_name,
                    torch_dtype="auto",
                    attn_implementation="flash_attention_2",
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_name,
                    torch_dtype="auto",
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
                if self.device == "cpu":
                    self.model = self.model.to(self.device)
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        self.model.eval()
        print(f"Model loaded successfully!")
    
    def generate(
        self,
        image: Image.Image,
        query: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = False
    ) -> Dict[str, str]:
        """
        Generate response for an image-query pair
        
        Args:
            image: PIL Image
            query: Text query
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            
        Returns:
            Dictionary containing generated response and metadata
        """
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": query},
                ],
            }
        ]
        
        # Prepare for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample
            )
        
        # Trim and decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return {
            'response': output_text,
            'model': self.model_name,
            'query': query
        }
    
    def batch_generate(
        self,
        samples: List[Dict],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = False,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Generate responses for a batch of samples
        
        Args:
            samples: List of samples, each containing 'image' and 'query'
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            show_progress: Whether to show progress bar
            
        Returns:
            List of dictionaries containing generated responses
        """
        results = []
        
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(samples, desc=f"Generating with {self.model_name}")
        else:
            iterator = samples
        
        for sample in iterator:
            try:
                result = self.generate(
                    image=sample['image'],
                    query=sample['query'],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample
                )
                result['id'] = sample.get('id', None)
                result['category'] = sample.get('category', 'Unknown')
                result['safeness_combination'] = sample.get('safeness_combination', 'Unknown')
                results.append(result)
            except Exception as e:
                print(f"Error processing sample {sample.get('id', 'unknown')}: {e}")
                results.append({
                    'response': f"ERROR: {str(e)}",
                    'model': self.model_name,
                    'query': sample['query'],
                    'id': sample.get('id', None),
                    'category': sample.get('category', 'Unknown'),
                    'safeness_combination': sample.get('safeness_combination', 'Unknown')
                })
        
        return results
    
    def clear_cache(self):
        """Clear CUDA cache to free memory"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    # Test the wrapper
    from PIL import Image
    import requests
    from io import BytesIO
    
    print("Testing Qwen-2.5-VL wrapper...")
    
    # Create a simple test image
    img = Image.new('RGB', (224, 224), color='red')
    
    # Test 3B model
    model = Qwen25VLWrapper(model_size="3B")
    result = model.generate(img, "What color is this image?")
    print(f"\nResponse: {result['response']}")
