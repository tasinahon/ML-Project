"""
Visualization and reporting utilities for evaluation results
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import numpy as np


class ResultsVisualizer:
    """Visualize and report evaluation results"""
    
    def __init__(self, results_dir: str = "./results"):
        """
        Initialize visualizer
        
        Args:
            results_dir: Directory containing evaluation results
        """
        self.results_dir = Path(results_dir)
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    
    def load_metrics(self, model_name: str) -> Dict:
        """Load metrics for a specific model"""
        metrics_file = self.results_dir / f"metrics_{model_name}.json"
        with open(metrics_file, 'r') as f:
            return json.load(f)
    
    def create_summary_table(self, metrics_files: List[str]) -> pd.DataFrame:
        """
        Create a summary table comparing multiple models
        
        Args:
            metrics_files: List of metrics file names
            
        Returns:
            DataFrame with comparison
        """
        data = []
        
        for metrics_file in metrics_files:
            with open(self.results_dir / metrics_file, 'r') as f:
                metrics = json.load(f)
            
            model_name = metrics_file.replace('metrics_', '').replace('.json', '')
            
            data.append({
                'Model': model_name,
                'Total Samples': metrics['total_samples'],
                'Refusal Rate (%)': round(metrics['refusal_rate'], 2),
                'Attack Success Rate (%)': round(metrics['attack_success_rate'], 2),
                'Total Refusals': metrics['total_refusals'],
                'Total Attacks': metrics['total_attacks_successful']
            })
        
        return pd.DataFrame(data)
    
    def plot_category_comparison(
        self,
        metrics: Dict,
        output_file: str = "category_comparison.png"
    ):
        """
        Plot category-wise metrics
        
        Args:
            metrics: Metrics dictionary
            output_file: Output file name
        """
        category_metrics = metrics['category_metrics']
        
        categories = list(category_metrics.keys())
        refusal_rates = [category_metrics[cat]['refusal_rate'] for cat in categories]
        asr_rates = [category_metrics[cat]['asr'] for cat in categories]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Refusal rates
        ax1.barh(categories, refusal_rates, color='steelblue')
        ax1.set_xlabel('Refusal Rate (%)', fontsize=12)
        ax1.set_title('Refusal Rate by Category', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, 100)
        
        # Add value labels
        for i, v in enumerate(refusal_rates):
            ax1.text(v + 1, i, f'{v:.1f}%', va='center')
        
        # Attack success rates
        ax2.barh(categories, asr_rates, color='coral')
        ax2.set_xlabel('Attack Success Rate (%)', fontsize=12)
        ax2.set_title('Attack Success Rate by Category', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 100)
        
        # Add value labels
        for i, v in enumerate(asr_rates):
            ax2.text(v + 1, i, f'{v:.1f}%', va='center')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / output_file, dpi=300, bbox_inches='tight')
        print(f"Saved category comparison to {self.results_dir / output_file}")
        plt.close()
    
    def plot_safeness_combination(
        self,
        metrics: Dict,
        output_file: str = "safeness_combination.png"
    ):
        """
        Plot safeness combination metrics
        
        Args:
            metrics: Metrics dictionary
            output_file: Output file name
        """
        safeness_metrics = metrics['safeness_metrics']
        
        combinations = list(safeness_metrics.keys())
        refusal_rates = [safeness_metrics[comb]['refusal_rate'] for comb in combinations]
        asr_rates = [safeness_metrics[comb]['asr'] for comb in combinations]
        totals = [safeness_metrics[comb]['total'] for comb in combinations]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Refusal rates
        axes[0, 0].bar(combinations, refusal_rates, color='steelblue', alpha=0.7)
        axes[0, 0].set_ylabel('Refusal Rate (%)', fontsize=11)
        axes[0, 0].set_title('Refusal Rate by Safeness Combination', fontsize=12, fontweight='bold')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_ylim(0, 100)
        
        # Add value labels
        for i, v in enumerate(refusal_rates):
            axes[0, 0].text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=9)
        
        # Attack success rates
        axes[0, 1].bar(combinations, asr_rates, color='coral', alpha=0.7)
        axes[0, 1].set_ylabel('Attack Success Rate (%)', fontsize=11)
        axes[0, 1].set_title('Attack Success Rate by Safeness Combination', fontsize=12, fontweight='bold')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].set_ylim(0, 100)
        
        # Add value labels
        for i, v in enumerate(asr_rates):
            axes[0, 1].text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=9)
        
        # Sample distribution
        axes[1, 0].bar(combinations, totals, color='forestgreen', alpha=0.7)
        axes[1, 0].set_ylabel('Number of Samples', fontsize=11)
        axes[1, 0].set_title('Sample Distribution by Safeness Combination', fontsize=12, fontweight='bold')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, v in enumerate(totals):
            axes[1, 0].text(i, v + max(totals)*0.02, str(v), ha='center', fontsize=9)
        
        # Stacked comparison
        x = np.arange(len(combinations))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, refusal_rates, width, label='Refusal Rate', color='steelblue', alpha=0.7)
        axes[1, 1].bar(x + width/2, asr_rates, width, label='Attack Success Rate', color='coral', alpha=0.7)
        axes[1, 1].set_ylabel('Rate (%)', fontsize=11)
        axes[1, 1].set_title('Comparison: Refusal vs Attack Success', fontsize=12, fontweight='bold')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(combinations, rotation=45, ha='right')
        axes[1, 1].legend()
        axes[1, 1].set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / output_file, dpi=300, bbox_inches='tight')
        print(f"Saved safeness combination plot to {self.results_dir / output_file}")
        plt.close()
    
    def generate_report(
        self,
        model_name: str,
        output_file: str = None
    ):
        """
        Generate a comprehensive text report
        
        Args:
            model_name: Name of the model
            output_file: Output file name (default: report_{model_name}.txt)
        """
        if output_file is None:
            output_file = f"report_{model_name}.txt"
        
        metrics = self.load_metrics(model_name)
        
        report = []
        report.append("=" * 80)
        report.append(f"EVALUATION REPORT: {model_name.upper()}")
        report.append("=" * 80)
        report.append("")
        
        # Overall metrics
        report.append("OVERALL METRICS")
        report.append("-" * 80)
        report.append(f"Total Samples Evaluated: {metrics['total_samples']}")
        report.append(f"Total Refusals: {metrics['total_refusals']}")
        report.append(f"Total Attack Successes: {metrics['total_attacks_successful']}")
        report.append(f"Refusal Rate: {metrics['refusal_rate']:.2f}%")
        report.append(f"Attack Success Rate: {metrics['attack_success_rate']:.2f}%")
        report.append("")
        
        # Category metrics
        report.append("CATEGORY-WISE METRICS")
        report.append("-" * 80)
        for category, cat_metrics in sorted(metrics['category_metrics'].items()):
            report.append(f"\n{category}:")
            report.append(f"  Total Samples: {cat_metrics['total']}")
            report.append(f"  Refusal Rate: {cat_metrics['refusal_rate']:.2f}%")
            report.append(f"  Attack Success Rate: {cat_metrics['asr']:.2f}%")
        report.append("")
        
        # Safeness combination metrics
        report.append("SAFENESS COMBINATION METRICS")
        report.append("-" * 80)
        for safeness, safe_metrics in sorted(metrics['safeness_metrics'].items()):
            report.append(f"\n{safeness}:")
            report.append(f"  Total Samples: {safe_metrics['total']}")
            report.append(f"  Refusal Rate: {safe_metrics['refusal_rate']:.2f}%")
            report.append(f"  Attack Success Rate: {safe_metrics['asr']:.2f}%")
        report.append("")
        
        # Save report
        report_text = "\n".join(report)
        with open(self.results_dir / output_file, 'w') as f:
            f.write(report_text)
        
        print(f"Generated report: {self.results_dir / output_file}")
        print(report_text)
        
        return report_text


def main():
    """Generate visualizations for existing results"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize evaluation results")
    parser.add_argument("--results_dir", type=str, default="./results", help="Results directory")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (e.g., qwen25vl_3b)")
    args = parser.parse_args()
    
    visualizer = ResultsVisualizer(args.results_dir)
    
    # Generate report
    visualizer.generate_report(args.model_name)
    
    # Load metrics
    metrics = visualizer.load_metrics(args.model_name)
    
    # Generate plots
    visualizer.plot_category_comparison(metrics, f"category_{args.model_name}.png")
    visualizer.plot_safeness_combination(metrics, f"safeness_{args.model_name}.png")
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
