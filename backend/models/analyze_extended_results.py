#!/usr/bin/env python3
"""
Analyze and summarize results from extended training experiments.
Generates comprehensive report for research paper.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """Analyze and summarize training results."""
    
    def __init__(self, results_dir: str = 'training_results_extended'):
        self.results_dir = Path(results_dir)
        self.all_results = []
        
    def load_all_results(self):
        """Load all result files."""
        logger.info("Loading all results...")
        
        result_files = list(self.results_dir.glob('results_*.json'))
        
        for result_file in result_files:
            with open(result_file, 'r') as f:
                result = json.load(f)
                self.all_results.append(result)
        
        logger.info(f"Loaded {len(self.all_results)} result files")
    
    def create_summary_table(self) -> pd.DataFrame:
        """Create summary table of all experiments."""
        summary_data = []
        
        for result in self.all_results:
            summary_data.append({
                'Experiment': result['experiment_name'],
                'Dataset': result['dataset'],
                'Model': result['model_type'],
                'Task': result['task'],
                'Num Classes': result['num_classes'],
                'Epochs Trained': len(result['training_history']['train_loss']),
                'Best Epoch': result['best_epoch'],
                'Best Val Acc': f"{result['best_val_acc']:.4f}",
                'Test Acc': f"{result['test_results']['accuracy']:.4f}",
                'Test Precision': f"{result['test_results']['precision']:.4f}",
                'Test Recall': f"{result['test_results']['recall']:.4f}",
                'Test F1': f"{result['test_results']['f1_score']:.4f}",
                'Training Time (min)': f"{result['training_time']/60:.2f}"
            })
        
        df = pd.DataFrame(summary_data)
        return df
    
    def create_comparison_plots(self):
        """Create comparison plots across experiments."""
        
        # 1. Test Accuracy Comparison
        fig, ax = plt.subplots(figsize=(15, 8))
        
        datasets = [r['dataset'] for r in self.all_results]
        models = [r['model_type'] for r in self.all_results]
        test_accs = [r['test_results']['accuracy'] for r in self.all_results]
        
        # Group by dataset
        dataset_groups = {}
        for i, dataset in enumerate(datasets):
            if dataset not in dataset_groups:
                dataset_groups[dataset] = {'models': [], 'accs': []}
            dataset_groups[dataset]['models'].append(models[i])
            dataset_groups[dataset]['accs'].append(test_accs[i])
        
        # Plot grouped bar chart
        x = np.arange(len(dataset_groups))
        width = 0.25
        
        for i, (dataset, data) in enumerate(dataset_groups.items()):
            model_types = set(data['models'])
            for j, model in enumerate(sorted(model_types)):
                model_accs = [data['accs'][k] for k in range(len(data['models'])) if data['models'][k] == model]
                offset = (j - len(model_types)/2 + 0.5) * width
                ax.bar(i + offset, np.mean(model_accs), width, label=model if i == 0 else "")
        
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel('Test Accuracy', fontsize=12)
        ax.set_title('Test Accuracy Comparison Across Datasets and Models', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_groups.keys())
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'comparison_test_accuracy.png', dpi=300, bbox_inches='tight')
        logger.info("Saved comparison plot: comparison_test_accuracy.png")
        plt.close()
        
        # 2. Training Convergence Comparison
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        datasets_unique = list(set([r['dataset'] for r in self.all_results]))
        
        for idx, dataset in enumerate(datasets_unique):
            ax = axes[idx]
            
            dataset_results = [r for r in self.all_results if r['dataset'] == dataset]
            
            for result in dataset_results:
                label = f"{result['model_type']}"
                val_acc = result['training_history']['val_acc']
                ax.plot(val_acc, label=label, linewidth=2)
            
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('Validation Accuracy', fontsize=11)
            ax.set_title(f'{dataset.upper()} - Validation Accuracy Over Time', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'comparison_convergence.png', dpi=300, bbox_inches='tight')
        logger.info("Saved comparison plot: comparison_convergence.png")
        plt.close()
        
        # 3. Model Performance Heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create matrix for heatmap
        datasets_unique = sorted(set([r['dataset'] for r in self.all_results]))
        models_unique = sorted(set([r['model_type'] for r in self.all_results]))
        
        heatmap_data = np.zeros((len(models_unique), len(datasets_unique)))
        
        for i, model in enumerate(models_unique):
            for j, dataset in enumerate(datasets_unique):
                matching_results = [r for r in self.all_results 
                                  if r['model_type'] == model and r['dataset'] == dataset]
                if matching_results:
                    heatmap_data[i, j] = matching_results[0]['test_results']['accuracy']
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlGnBu',
                   xticklabels=datasets_unique, yticklabels=models_unique,
                   cbar_kws={'label': 'Test Accuracy'}, ax=ax)
        ax.set_title('Model Performance Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel('Model Architecture', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'heatmap_performance.png', dpi=300, bbox_inches='tight')
        logger.info("Saved heatmap: heatmap_performance.png")
        plt.close()
    
    def generate_latex_tables(self) -> Dict[str, str]:
        """Generate LaTeX tables for research paper."""
        
        latex_tables = {}
        
        # Main results table
        latex_main = "\\begin{table}[H]\n"
        latex_main += "\\centering\n"
        latex_main += "\\caption{Extended Training Results (50-100 Epochs)}\n"
        latex_main += "\\begin{tabular}{@{}llllll@{}}\n"
        latex_main += "\\toprule\n"
        latex_main += "\\textbf{Dataset} & \\textbf{Model} & \\textbf{Epochs} & \\textbf{Val Acc} & \\textbf{Test Acc} & \\textbf{F1 Score} \\\\ \\midrule\n"
        
        for result in sorted(self.all_results, key=lambda x: (x['dataset'], x['model_type'])):
            dataset = result['dataset'].replace('mnist', 'MNIST')
            model = result['model_type'].capitalize()
            epochs = len(result['training_history']['train_loss'])
            val_acc = f"{result['best_val_acc']*100:.2f}\\%"
            test_acc = f"{result['test_results']['accuracy']*100:.2f}\\%"
            f1 = f"{result['test_results']['f1_score']:.3f}"
            
            latex_main += f"{dataset} & {model} & {epochs} & {val_acc} & {test_acc} & {f1} \\\\\n"
        
        latex_main += "\\bottomrule\n"
        latex_main += "\\end{tabular}\n"
        latex_main += "\\end{table}\n"
        
        latex_tables['main_results'] = latex_main
        
        # Save LaTeX tables
        with open(self.results_dir / 'latex_tables.tex', 'w') as f:
            for name, table in latex_tables.items():
                f.write(f"% {name}\n")
                f.write(table)
                f.write("\n\n")
        
        logger.info("Generated LaTeX tables: latex_tables.tex")
        
        return latex_tables
    
    def generate_research_paper_summary(self):
        """Generate comprehensive summary for research paper."""
        
        summary = {
            'total_experiments': len(self.all_results),
            'datasets': list(set([r['dataset'] for r in self.all_results])),
            'models': list(set([r['model_type'] for r in self.all_results])),
            'total_training_time_hours': sum([r['training_time'] for r in self.all_results]) / 3600,
            'best_results_by_dataset': {}
        }
        
        # Find best results per dataset
        for dataset in summary['datasets']:
            dataset_results = [r for r in self.all_results if r['dataset'] == dataset]
            best_result = max(dataset_results, key=lambda x: x['test_results']['accuracy'])
            
            summary['best_results_by_dataset'][dataset] = {
                'model': best_result['model_type'],
                'test_accuracy': best_result['test_results']['accuracy'],
                'test_f1': best_result['test_results']['f1_score'],
                'epochs': len(best_result['training_history']['train_loss']),
                'best_epoch': best_result['best_epoch']
            }
        
        # Calculate statistics
        all_test_accs = [r['test_results']['accuracy'] for r in self.all_results]
        summary['statistics'] = {
            'mean_test_accuracy': np.mean(all_test_accs),
            'std_test_accuracy': np.std(all_test_accs),
            'min_test_accuracy': np.min(all_test_accs),
            'max_test_accuracy': np.max(all_test_accs)
        }
        
        # Save summary
        with open(self.results_dir / 'research_paper_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Generated research paper summary: research_paper_summary.json")
        
        return summary
    
    def generate_markdown_report(self):
        """Generate markdown report."""
        
        report = "# Extended Training Results - Research Paper\n\n"
        report += f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += f"**Total Experiments**: {len(self.all_results)}\n\n"
        
        # Summary table
        df = self.create_summary_table()
        report += "## Summary Table\n\n"
        report += df.to_markdown(index=False)
        report += "\n\n"
        
        # Best results per dataset
        report += "## Best Results by Dataset\n\n"
        
        datasets = sorted(set([r['dataset'] for r in self.all_results]))
        for dataset in datasets:
            dataset_results = [r for r in self.all_results if r['dataset'] == dataset]
            best_result = max(dataset_results, key=lambda x: x['test_results']['accuracy'])
            
            report += f"### {dataset.upper()}\n\n"
            report += f"- **Best Model**: {best_result['model_type']}\n"
            report += f"- **Test Accuracy**: {best_result['test_results']['accuracy']:.4f}\n"
            report += f"- **Test F1-Score**: {best_result['test_results']['f1_score']:.4f}\n"
            report += f"- **Epochs Trained**: {len(best_result['training_history']['train_loss'])}\n"
            report += f"- **Best Epoch**: {best_result['best_epoch']}\n"
            report += f"- **Training Time**: {best_result['training_time']/60:.2f} minutes\n\n"
        
        # Model comparison
        report += "## Model Architecture Comparison\n\n"
        
        models = sorted(set([r['model_type'] for r in self.all_results]))
        for model in models:
            model_results = [r for r in self.all_results if r['model_type'] == model]
            test_accs = [r['test_results']['accuracy'] for r in model_results]
            
            report += f"### {model.capitalize()}\n\n"
            report += f"- **Mean Test Accuracy**: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}\n"
            report += f"- **Best Accuracy**: {np.max(test_accs):.4f}\n"
            report += f"- **Experiments**: {len(model_results)}\n\n"
        
        # Save report
        with open(self.results_dir / 'RESULTS_REPORT.md', 'w') as f:
            f.write(report)
        
        logger.info("Generated markdown report: RESULTS_REPORT.md")
    
    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        logger.info("="*50)
        logger.info("Starting Results Analysis")
        logger.info("="*50)
        
        # Load results
        self.load_all_results()
        
        if not self.all_results:
            logger.error("No results found to analyze!")
            return
        
        # Create summary table
        logger.info("\nCreating summary table...")
        df = self.create_summary_table()
        df.to_csv(self.results_dir / 'summary_table.csv', index=False)
        logger.info("Saved: summary_table.csv")
        
        # Create comparison plots
        logger.info("\nCreating comparison plots...")
        self.create_comparison_plots()
        
        # Generate LaTeX tables
        logger.info("\nGenerating LaTeX tables...")
        self.generate_latex_tables()
        
        # Generate research paper summary
        logger.info("\nGenerating research paper summary...")
        self.generate_research_paper_summary()
        
        # Generate markdown report
        logger.info("\nGenerating markdown report...")
        self.generate_markdown_report()
        
        logger.info("\n" + "="*50)
        logger.info("Analysis completed successfully!")
        logger.info("="*50)
        logger.info(f"\nAll results saved in: {self.results_dir}/")


def main():
    analyzer = ResultsAnalyzer()
    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()



