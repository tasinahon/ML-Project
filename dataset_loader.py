"""
Dataset loader for HoliSafe-Bench
Loads the holistic safety benchmark for Vision-Language Models
"""

from datasets import load_dataset
from typing import Dict, List, Optional
import json
from PIL import Image
from huggingface_hub import hf_hub_download


class HoliSafeBenchLoader:
    """Loader for HoliSafe-Bench dataset"""
    
    def __init__(self, use_hf_api: bool = True):
        """
        Initialize the dataset loader
        
        Args:
            use_hf_api: If True, use HuggingFace Datasets API. Otherwise, use direct file access.
        """
        self.use_hf_api = use_hf_api
        self.dataset = None
        self.data = None
        
    def load(self):
        """Load the HoliSafe-Bench dataset"""
        print("Loading HoliSafe-Bench dataset...")
        
        if self.use_hf_api:
            # Option 1: Using Hugging Face Datasets API (Recommended)
            self.dataset = load_dataset("etri-vilab/holisafe-bench")
            print(f"Loaded dataset with {len(self.dataset['test'])} samples")
        else:
            # Option 2: Direct File Access
            json_path = hf_hub_download(
                repo_id="etri-vilab/holisafe-bench",
                filename="holisafe_bench.json",
                repo_type="dataset"
            )
            
            with open(json_path, 'r') as f:
                self.data = json.load(f)
            print(f"Loaded dataset with {len(self.data)} samples")
    
    def get_sample(self, idx: int) -> Dict:
        """
        Get a sample from the dataset
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing sample data
        """
        if self.use_hf_api:
            sample = self.dataset['test'][idx]
            return {
                'id': sample.get('id', idx),
                'image': sample['image'],
                'query': sample['query'],
                'category': sample.get('category', 'Unknown'),
                'subcategory': sample.get('subcategory', 'Unknown'),
                'safeness_combination': sample.get('safeness_combination', 'Unknown'),
                'expected_response': sample.get('expected_response', ''),
                'is_safe': sample.get('is_safe', None)
            }
        else:
            sample = self.data[idx]
            # Download image
            image_path = hf_hub_download(
                repo_id="etri-vilab/holisafe-bench",
                filename=f"images/{sample['image']}",
                repo_type="dataset"
            )
            image = Image.open(image_path)
            
            return {
                'id': sample.get('id', idx),
                'image': image,
                'query': sample['query'],
                'category': sample.get('category', 'Unknown'),
                'subcategory': sample.get('subcategory', 'Unknown'),
                'safeness_combination': sample.get('safeness_combination', 'Unknown'),
                'expected_response': sample.get('expected_response', ''),
                'is_safe': sample.get('is_safe', None)
            }
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset"""
        if self.use_hf_api:
            return len(self.dataset['test'])
        else:
            return len(self.data)
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        stats = {
            'total_samples': len(self),
            'categories': {},
            'safeness_combinations': {}
        }
        
        for i in range(len(self)):
            sample = self.get_sample(i)
            
            # Count categories
            category = sample['category']
            stats['categories'][category] = stats['categories'].get(category, 0) + 1
            
            # Count safeness combinations
            safeness = sample['safeness_combination']
            stats['safeness_combinations'][safeness] = stats['safeness_combinations'].get(safeness, 0) + 1
        
        return stats


if __name__ == "__main__":
    # Test the loader
    loader = HoliSafeBenchLoader(use_hf_api=True)
    loader.load()
    
    print("\n=== Dataset Statistics ===")
    stats = loader.get_statistics()
    print(f"Total samples: {stats['total_samples']}")
    print(f"\nCategories: {stats['categories']}")
    print(f"\nSafeness combinations: {stats['safeness_combinations']}")
    
    print("\n=== Sample Data ===")
    sample = loader.get_sample(0)
    print(f"ID: {sample['id']}")
    print(f"Query: {sample['query']}")
    print(f"Category: {sample['category']}")
    print(f"Subcategory: {sample['subcategory']}")
    print(f"Safeness: {sample['safeness_combination']}")
    print(f"Image size: {sample['image'].size}")
