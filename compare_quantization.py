"""
Quantization Safety Drift Analysis
Compares FP16 baseline with quantized models to measure safety degradation
For the project: "Quantization-Induced Safety Drift in Multimodal LLMs"
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, List


class QuantizationSafetyAnalyzer:
    """Analyze safety drift from quantization"""
    
    def __init__(self, results_dir: str = "./results"):
        self.results_dir = Path(results_dir)
        
    def load_metrics(self, model_size: str, precision: str) -> Dict:
        """
        Load metrics for a specific configuration
        
        Args:
            model_size: "3b" or "7b"
            precision: "fp16", "bitsandbytes4bit", "bitsandbytes8bit", etc.
        """
        metrics_file = self.results_dir / f"metrics_qwen25vl_{model_size}_{precision}.json"
        
        if not metrics_file.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
        
        with open(metrics_file, 'r') as f:
            return json.load(f)
    
    def compare_safety_drift(
        self,
        model_size: str,
        baseline: str = "fp16",
        quantized: List[str] = ["bitsandbytes4bit"]
    ) -> pd.DataFrame:
        """
        Compare safety metrics between baseline and quantized models
        
        Args:
            model_size: "3b" or "7b"
            baseline: Baseline precision (usually "fp16")
            quantized: List of quantized configurations to compare
            
        Returns:
            DataFrame with comparison results
        """
        # Load baseline
        baseline_metrics = self.load_metrics(model_size, baseline)
        
        results = []
        
        # Baseline row
        results.append({
            'Configuration': f'{model_size.upper()}-{baseline.upper()}',
            'Precision': baseline,
            'Refusal Rate (%)': baseline_metrics['refusal_rate'],
            'ASR (%)': baseline_metrics['attack_success_rate'],
            'Total Refusals': baseline_metrics['total_refusals'],
            'Δ Refusal Rate': 0.0,
            'Δ ASR': 0.0
        })
        
        # Quantized models
        for quant in quantized:
            try:
                quant_metrics = self.load_metrics(model_size, quant)
                
                delta_rr = quant_metrics['refusal_rate'] - baseline_metrics['refusal_rate']
                delta_asr = quant_metrics['attack_success_rate'] - baseline_metrics['attack_success_rate']
                
                results.append({
                    'Configuration': f'{model_size.upper()}-{quant.upper()}',
                    'Precision': quant,
                    'Refusal Rate (%)': quant_metrics['refusal_rate'],
                    'ASR (%)': quant_metrics['attack_success_rate'],
                    'Total Refusals': quant_metrics['total_refusals'],
                    'Δ Refusal Rate': delta_rr,
                    'Δ ASR': delta_asr
                })
            except FileNotFoundError:
                print(f"Warning: Metrics not found for {model_size}-{quant}")
        
        return pd.DataFrame(results)
    
    def analyze_category_drift(
        self,
        model_size: str,
        baseline: str = "fp16",
        quantized: str = "bitsandbytes4bit"
    ) -> pd.DataFrame:
        """
        Analyze which safety categories degrade most under quantization
        
        This addresses H1: "Fragile Guardrail" - some categories may be more fragile
        """
        baseline_metrics = self.load_metrics(model_size, baseline)
        quant_metrics = self.load_metrics(model_size, quantized)
        
        results = []
        
        for category in baseline_metrics['category_metrics'].keys():
            baseline_cat = baseline_metrics['category_metrics'][category]
            quant_cat = quant_metrics['category_metrics'][category]
            
            results.append({
                'Category': category,
                'Baseline RR (%)': baseline_cat['refusal_rate'],
                'Quantized RR (%)': quant_cat['refusal_rate'],
                'Δ RR': quant_cat['refusal_rate'] - baseline_cat['refusal_rate'],
                'Baseline ASR (%)': baseline_cat['asr'],
                'Quantized ASR (%)': quant_cat['asr'],
                'Δ ASR': quant_cat['asr'] - baseline_cat['asr'],
                'Samples': baseline_cat['total']
            })
        
        df = pd.DataFrame(results)
        return df.sort_values('Δ ASR', ascending=False)
    
    def plot_safety_drift(
        self,
        comparison_df: pd.DataFrame,
        output_file: str = "safety_drift_comparison.png"
    ):
        """
        Visualize safety drift from quantization
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        configs = comparison_df['Configuration']
        
        # Plot 1: Refusal Rate comparison
        ax1.bar(configs, comparison_df['Refusal Rate (%)'], 
                color=['steelblue' if 'FP16' in c else 'coral' for c in configs],
                alpha=0.7)
        ax1.set_ylabel('Refusal Rate (%)', fontsize=12)
        ax1.set_title('Refusal Rate: FP16 vs Quantized', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.axhline(y=comparison_df.iloc[0]['Refusal Rate (%)'], 
                    color='green', linestyle='--', label='FP16 Baseline')
        ax1.legend()
        
        # Plot 2: ASR comparison
        ax2.bar(configs, comparison_df['ASR (%)'], 
                color=['steelblue' if 'FP16' in c else 'coral' for c in configs],
                alpha=0.7)
        ax2.set_ylabel('Attack Success Rate (%)', fontsize=12)
        ax2.set_title('ASR: FP16 vs Quantized', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.axhline(y=comparison_df.iloc[0]['ASR (%)'], 
                    color='red', linestyle='--', label='FP16 Baseline')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.results_dir / output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {self.results_dir / output_file}")
        plt.close()
    
    def plot_category_drift(
        self,
        category_df: pd.DataFrame,
        output_file: str = "category_drift.png"
    ):
        """
        Visualize which categories are most affected by quantization
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        categories = category_df['Category']
        x = np.arange(len(categories))
        width = 0.35
        
        ax.bar(x - width/2, category_df['Δ RR'], width, 
               label='Δ Refusal Rate', color='steelblue', alpha=0.7)
        ax.bar(x + width/2, category_df['Δ ASR'], width, 
               label='Δ Attack Success Rate', color='coral', alpha=0.7)
        
        ax.set_xlabel('Safety Category', fontsize=12)
        ax.set_ylabel('Change (%)', fontsize=12)
        ax.set_title('Safety Drift by Category: Quantization Impact', 
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {self.results_dir / output_file}")
        plt.close()
    
    def generate_research_report(
        self,
        model_size: str,
        baseline: str = "fp16",
        quantized: str = "bitsandbytes4bit"
    ):
        """
        Generate a comprehensive research report for the quantization safety project
        """
        print("="*80)
        print("QUANTIZATION SAFETY DRIFT ANALYSIS")
        print("Project: Quantization-Induced Safety Drift in Multimodal LLMs")
        print("="*80)
        print()
        
        # Overall comparison
        comparison = self.compare_safety_drift(model_size, baseline, [quantized])
        print("📊 OVERALL SAFETY METRICS")
        print("-"*80)
        print(comparison.to_string(index=False))
        print()
        
        # Calculate drift magnitude
        drift = comparison.iloc[1]
        print("🔍 SAFETY DRIFT ANALYSIS")
        print("-"*80)
        print(f"Refusal Rate Change: {drift['Δ Refusal Rate']:+.2f}%")
        print(f"ASR Change: {drift['Δ ASR']:+.2f}%")
        
        if drift['Δ ASR'] > 5:
            print("⚠️  SIGNIFICANT SAFETY DEGRADATION DETECTED!")
            print(f"   Quantization increased attack success by {drift['Δ ASR']:.1f}%")
        elif drift['Δ ASR'] > 0:
            print("⚡ Moderate safety degradation observed")
        else:
            print("✅ No significant safety degradation")
        print()
        
        # Category analysis
        print("📋 CATEGORY-WISE DRIFT (Hypothesis H1: Fragile Guardrails)")
        print("-"*80)
        category_drift = self.analyze_category_drift(model_size, baseline, quantized)
        print(category_drift.to_string(index=False))
        print()
        
        # Identify most fragile categories
        most_affected = category_drift.nlargest(3, 'Δ ASR')
        print("⚠️  MOST FRAGILE SAFETY CATEGORIES:")
        for _, row in most_affected.iterrows():
            print(f"   {row['Category']}: ASR +{row['Δ ASR']:.1f}% (RR {row['Δ RR']:+.1f}%)")
        print()
        
        # Research implications
        print("🔬 RESEARCH IMPLICATIONS")
        print("-"*80)
        print("H1 (Fragile Guardrail): Category-specific drift suggests some safety")
        print("    circuits are more vulnerable to quantization than others.")
        print()
        print("H2 (Mechanism Drift): ASR increase indicates primary refusal circuits")
        print("    may be degrading, forcing model to use weaker secondary pathways.")
        print()
        
        # Generate visualizations
        self.plot_safety_drift(comparison, f"safety_drift_{model_size}.png")
        self.plot_category_drift(category_drift, f"category_drift_{model_size}.png")
        
        print("="*80)
        print("Report complete! Check results directory for visualizations.")
        print("="*80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare FP16 baseline with quantized models"
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="7b",
        choices=["3b", "7b"],
        help="Model size"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="fp16",
        help="Baseline precision"
    )
    parser.add_argument(
        "--quantized",
        type=str,
        default="bitsandbytes4bit",
        help="Quantized configuration to compare"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results",
        help="Results directory"
    )
    
    args = parser.parse_args()
    
    analyzer = QuantizationSafetyAnalyzer(args.results_dir)
    analyzer.generate_research_report(
        args.model_size,
        args.baseline,
        args.quantized
    )


if __name__ == "__main__":
    main()
