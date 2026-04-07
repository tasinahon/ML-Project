"""
SafeQwen2.5-VL model wrapper with safety classification support
Handles both full precision and quantized versions
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from typing import Dict, List, Optional
from PIL import Image
import gc


class SafeQwen25VLWrapper:
    """Wrapper for SafeQwen2.5-VL with safety classification"""
    
    def __init__(
        self, 
        model_size: str = "7B",
        device: str = "auto",
        quantization: Optional[str] = None,
        bits: int = 4,
        quant_type: str = "nf4",
        enable_safety_classifier: bool = True,
        quant_scope: str = "all"
    ):
        """
        Initialize SafeQwen2.5-VL model
        
        Args:
            model_size: Model size ("7B")
            device: Device to run on
            quantization: "bitsandbytes" or None for FP16
            bits: 4 or 8 for quantization
            quant_type: "nf4" or "fp4" for 4-bit quantization
            enable_safety_classifier: Whether to use safety classification
            quant_scope: "all", "vision_only", or "llm_only"
        """
        self.model_size = model_size
        self.quantization = quantization
        self.bits = bits
        self.quant_type = quant_type
        self.enable_safety_classifier = enable_safety_classifier
        self.quant_scope = quant_scope

        if self.quant_scope not in ["all", "vision_only", "llm_only"]:
            raise ValueError(
                f"quant_scope must be 'all', 'vision_only', or 'llm_only', got {self.quant_scope}"
            )
        
        # Model names
        self.model_name = f"etri-vilab/SafeQwen2.5-VL-{model_size}"
        self.processor_name = f"Qwen/Qwen2.5-VL-{model_size}-Instruct"
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        precision_str = f"{quantization}-{bits}bit" if quantization else "FP16"
        if quantization == "bitsandbytes" and self.quant_scope == "vision_only":
            precision_str += " (vision-only)"
        elif quantization == "bitsandbytes" and self.quant_scope == "llm_only":
            precision_str += " (llm-only)"
        print(f"Loading {self.model_name} on {self.device} ({precision_str})...")
        
        # Load model with quantization if specified
        if quantization == "bitsandbytes":
            from transformers import BitsAndBytesConfig

            skip_modules = None
            if self.quant_scope == "vision_only":
                skip_modules = self._build_non_vision_skip_modules()
                print(f"Using selective quantization: quantizing vision modules only")
                print(f"Skipping {len(skip_modules)} non-vision modules from quantization")
            elif self.quant_scope == "llm_only":
                skip_modules = self._build_vision_skip_modules()
                print(f"Using selective quantization: quantizing LLM modules only")
                print(f"Skipping {len(skip_modules)} vision modules from quantization")
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=(bits == 4),
                load_in_8bit=(bits == 8),
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type=quant_type,  # "nf4" or "fp4"
                llm_int8_skip_modules=skip_modules
            )
            
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            # FP16 baseline
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.processor_name)
        
        # Get safety categories if available
        self.safety_categories = getattr(self.model.config, 'safety_categories', None)
        
        self.model.eval()
        print(f"Model loaded successfully!")
        if self.safety_categories:
            print(f"Safety categories available: {len(self.safety_categories)}")

    @staticmethod
    def _is_vision_module(module_name: str) -> bool:
        """Heuristic matcher for vision module prefixes in Qwen2.5-VL/SafeQwen."""
        if not module_name:
            return False

        vision_prefixes = (
            "visual",
            "model.visual",
            "vision_tower",
            "model.vision_tower",
        )
        return module_name.startswith(vision_prefixes)

    def _build_non_vision_skip_modules(self) -> List[str]:
        """
        Build a skip list so bitsandbytes keeps non-vision modules in full precision.

        This performs a temporary CPU load to inspect module names, then unloads it.
        """
        print("Building non-vision skip-module list (one-time module scan)...")

        temp_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            device_map="cpu",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        skip_modules = []
        for module_name, _ in temp_model.named_modules():
            if not module_name:
                continue
            if not self._is_vision_module(module_name):
                skip_modules.append(module_name)

        del temp_model
        gc.collect()

        if len(skip_modules) == 0:
            raise RuntimeError(
                "Failed to build non-vision skip list; refusing to quantize all modules in vision-only mode."
            )

        return skip_modules

    def _build_vision_skip_modules(self) -> List[str]:
        """
        Build a skip list so bitsandbytes keeps vision modules in full precision.
        """
        print("Building vision skip-module list (one-time module scan)...")

        temp_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            device_map="cpu",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        skip_modules = []
        for module_name, _ in temp_model.named_modules():
            if not module_name:
                continue
            if self._is_vision_module(module_name):
                skip_modules.append(module_name)

        del temp_model
        gc.collect()

        if len(skip_modules) == 0:
            raise RuntimeError(
                "Failed to build vision skip list; refusing to run llm-only quantization without protected vision modules."
            )

        return skip_modules
    
    def generate(
        self,
        image: Image.Image,
        query: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = False
    ) -> Dict[str, any]:
        """
        Generate response with safety classification
        
        Returns:
            Dictionary with response, safety scores, and metadata
        """
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": query},
                ],
            }
        ]
        
        # Process inputs
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
        ).to(self.device)
        
        # Get safety classification if enabled
        safety_scores = None
        if self.enable_safety_classifier:
            try:
                with torch.no_grad():
                    outputs = self.model(**inputs, do_safety=True)
                    if hasattr(outputs, 'img_safety_probs'):
                        safety_probs = outputs.img_safety_probs[0].cpu().numpy()
                        safety_scores = {
                            category: float(prob)
                            for category, prob in zip(self.safety_categories, safety_probs)
                            if prob > 0.01  # Only include >1% probability
                        }
            except Exception as e:
                print(f"Warning: Safety classification failed: {e}")
                safety_scores = None
        
        # Generate text response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample
            )
        
        # Clear memory immediately after generation
        torch.cuda.empty_cache()
        
        # Decode output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        # Clean up
        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()
        
        return {
            'response': output_text,
            'safety_scores': safety_scores,
            'model': self.model_name,
            'query': query
        }
    
    def batch_generate(
        self,
        samples: List[Dict],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = False,
        show_progress: bool = True,
        memory_cleanup_interval: int = 10
    ) -> List[Dict]:
        """
        Generate responses for multiple samples with memory management
        
        Args:
            memory_cleanup_interval: Clean CUDA cache every N samples
        """
        results = []
        
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(samples, desc=f"Generating with {self.model_name}")
        else:
            iterator = samples
        
        for idx, sample in enumerate(iterator):
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
                
                # Periodic memory cleanup
                if (idx + 1) % memory_cleanup_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                    
            except Exception as e:
                print(f"Error processing sample {sample.get('id', 'unknown')}: {e}")
                results.append({
                    'response': f"ERROR: {str(e)}",
                    'safety_scores': None,
                    'model': self.model_name,
                    'query': sample['query'],
                    'id': sample.get('id', None),
                    'category': sample.get('category', 'Unknown'),
                    'safeness_combination': sample.get('safeness_combination', 'Unknown')
                })
                
                # Clean up on error
                torch.cuda.empty_cache()
                gc.collect()
        
        return results
    
    def clear_cache(self):
        """Aggressive memory cleanup"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
