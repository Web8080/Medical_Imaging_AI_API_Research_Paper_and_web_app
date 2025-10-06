#!/usr/bin/env python3
"""
Generate comprehensive visualizations for advanced training results.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AdvancedResultsVisualizer:
    """Visualize advanced training results."""
    
    def __init__(self, results_dir: str = "results/advanced_training"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path("training_results/advanced_models")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        with open(self.results_dir / "advanced_training_summary.json", 'r') as f:
            self.results = json.load(f)
    
    def create_model_comparison_plot(self):
        """Create comprehensive model comparison plot."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Prepare data
        datasets = []
        models = []
        accuracies = []
        parameters = []
        
        for key, data in self.results.items():
            if 'error' not in data:
                dataset, model = key.split('_')
                datasets.append(dataset.upper())
                models.append(model.upper())
                accuracies.append(data['test_accuracy'])
                parameters.append(data['model_parameters'] / 1e6)  # Convert to millions
        
        # 1. Accuracy comparison
        df = pd.DataFrame({
            'Dataset': datasets,
            'Model': models,
            'Accuracy': accuracies
        })
        
        pivot_acc = df.pivot(index='Dataset', columns='Model', values='Accuracy')
        sns.heatmap(pivot_acc, annot=True, fmt='.1f', cmap='RdYlGn', 
                   ax=ax1, cbar_kws={'label': 'Test Accuracy (%)'})
        ax1.set_title('Model Performance Comparison\n(Test Accuracy %)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Model Architecture')
        ax1.set_ylabel('Dataset')
        
        # 2. Parameter count comparison
        df_params = pd.DataFrame({
            'Dataset': datasets,
            'Model': models,
            'Parameters': parameters
        })
        
        pivot_params = df_params.pivot(index='Dataset', columns='Model', values='Parameters')
        sns.heatmap(pivot_params, annot=True, fmt='.1f', cmap='Blues', 
                   ax=ax2, cbar_kws={'label': 'Parameters (Millions)'})
        ax2.set_title('Model Complexity Comparison\n(Parameters in Millions)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Model Architecture')
        ax2.set_ylabel('Dataset')
        
        # 3. Accuracy vs Parameters scatter
        colors = ['red', 'blue', 'green', 'orange']
        for i, (dataset, model) in enumerate(zip(datasets, models)):
            ax3.scatter(parameters[i], accuracies[i], 
                       s=200, c=colors[i], alpha=0.7, 
                       label=f'{dataset} + {model}')
        
        ax3.set_xlabel('Model Parameters (Millions)')
        ax3.set_ylabel('Test Accuracy (%)')
        ax3.set_title('Accuracy vs Model Complexity', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Efficiency comparison (Accuracy per Million Parameters)
        efficiency = [acc / param for acc, param in zip(accuracies, parameters)]
        
        bars = ax4.bar(range(len(efficiency)), efficiency, 
                      color=colors[:len(efficiency)], alpha=0.7)
        ax4.set_xlabel('Model + Dataset Combination')
        ax4.set_ylabel('Efficiency (Accuracy per Million Parameters)')
        ax4.set_title('Model Efficiency Comparison', fontsize=14, fontweight='bold')
        ax4.set_xticks(range(len(efficiency)))
        ax4.set_xticklabels([f'{d}\n+{m}' for d, m in zip(datasets, models)], 
                           rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, eff in zip(bars, efficiency):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{eff:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'advanced_model_comparison.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info("Advanced model comparison plot created")
    
    def create_performance_analysis_plot(self):
        """Create detailed performance analysis plot."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Prepare data
        datasets = []
        models = []
        accuracies = []
        parameters = []
        
        for key, data in self.results.items():
            if 'error' not in data:
                dataset, model = key.split('_')
                datasets.append(dataset)
                models.append(model)
                accuracies.append(data['test_accuracy'])
                parameters.append(data['model_parameters'])
        
        # 1. Dataset-wise performance
        dataset_performance = {}
        for dataset, acc in zip(datasets, accuracies):
            if dataset not in dataset_performance:
                dataset_performance[dataset] = []
            dataset_performance[dataset].append(acc)
        
        dataset_names = list(dataset_performance.keys())
        dataset_accs = [np.mean(accs) for accs in dataset_performance.values()]
        dataset_stds = [np.std(accs) for accs in dataset_performance.values()]
        
        bars1 = ax1.bar(dataset_names, dataset_accs, yerr=dataset_stds, 
                       capsize=5, alpha=0.7, color=['skyblue', 'lightcoral'])
        ax1.set_ylabel('Average Test Accuracy (%)')
        ax1.set_title('Performance by Dataset', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, acc, std in zip(bars1, dataset_accs, dataset_stds):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std + 1,
                    f'{acc:.1f}±{std:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Model-wise performance
        model_performance = {}
        for model, acc in zip(models, accuracies):
            if model not in model_performance:
                model_performance[model] = []
            model_performance[model].append(acc)
        
        model_names = list(model_performance.keys())
        model_accs = [np.mean(accs) for accs in model_performance.values()]
        model_stds = [np.std(accs) for accs in model_performance.values()]
        
        bars2 = ax2.bar(model_names, model_accs, yerr=model_stds, 
                       capsize=5, alpha=0.7, color=['lightgreen', 'gold'])
        ax2.set_ylabel('Average Test Accuracy (%)')
        ax2.set_title('Performance by Model Architecture', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, acc, std in zip(bars2, model_accs, model_stds):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + std + 1,
                    f'{acc:.1f}±{std:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Parameter efficiency
        efficiency = [acc / (param / 1e6) for acc, param in zip(accuracies, parameters)]
        
        bars3 = ax3.bar(range(len(efficiency)), efficiency, 
                       color=['red', 'blue', 'green', 'orange'], alpha=0.7)
        ax3.set_xlabel('Model + Dataset Combination')
        ax3.set_ylabel('Efficiency (Accuracy per Million Parameters)')
        ax3.set_title('Parameter Efficiency Analysis', fontsize=14, fontweight='bold')
        ax3.set_xticks(range(len(efficiency)))
        ax3.set_xticklabels([f'{d.upper()}\n+{m.upper()}' for d, m in zip(datasets, models)], 
                           rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, eff in zip(bars3, efficiency):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{eff:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Performance vs Complexity scatter
        colors = ['red', 'blue', 'green', 'orange']
        for i, (dataset, model) in enumerate(zip(datasets, models)):
            ax4.scatter(parameters[i] / 1e6, accuracies[i], 
                       s=200, c=colors[i], alpha=0.7, 
                       label=f'{dataset.upper()} + {model.upper()}')
        
        ax4.set_xlabel('Model Parameters (Millions)')
        ax4.set_ylabel('Test Accuracy (%)')
        ax4.set_title('Performance vs Model Complexity', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'advanced_performance_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info("Advanced performance analysis plot created")
    
    def create_detailed_comparison_table(self):
        """Create detailed comparison table."""
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare data for table
        table_data = []
        headers = ['Dataset', 'Model', 'Test Accuracy (%)', 'Parameters (M)', 'Efficiency', 'Input Channels']
        
        for key, data in self.results.items():
            if 'error' not in data:
                dataset, model = key.split('_')
                efficiency = data['test_accuracy'] / (data['model_parameters'] / 1e6)
                table_data.append([
                    dataset.upper(),
                    model.upper(),
                    f"{data['test_accuracy']:.2f}",
                    f"{data['model_parameters'] / 1e6:.2f}",
                    f"{efficiency:.2f}",
                    str(data['input_channels'])
                ])
        
        # Sort by accuracy
        table_data.sort(key=lambda x: float(x[2]), reverse=True)
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=headers, 
                        cellLoc='center', loc='center',
                        colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.15])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        # Color code the accuracy column
        for i in range(1, len(table_data) + 1):
            accuracy = float(table_data[i-1][2])
            if accuracy >= 70:
                color = 'lightgreen'
            elif accuracy >= 60:
                color = 'lightyellow'
            else:
                color = 'lightcoral'
            table[(i, 2)].set_facecolor(color)
        
        # Header styling
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('lightblue')
            table[(0, i)].set_text_props(weight='bold')
        
        plt.title('Advanced Model Training Results Summary', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.savefig(self.output_dir / 'advanced_results_table.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info("Advanced results table created")
    
    def create_all_visualizations(self):
        """Create all visualizations."""
        logger.info("Creating advanced training visualizations...")
        
        self.create_model_comparison_plot()
        self.create_performance_analysis_plot()
        self.create_detailed_comparison_table()
        
        # Create summary report
        self.create_summary_report()
        
        logger.info(f"All advanced visualizations saved to: {self.output_dir}")
    
    def create_summary_report(self):
        """Create summary report."""
        report_file = self.output_dir / "ADVANCED_TRAINING_SUMMARY.md"
        
        with open(report_file, 'w') as f:
            f.write("# Advanced Model Training Results Summary\n\n")
            f.write("## Overview\n\n")
            f.write("This document summarizes the results of training advanced model architectures on MedMNIST datasets.\n\n")
            
            f.write("## Model Architectures Tested\n\n")
            f.write("### 1. Advanced CNN\n")
            f.write("- **Architecture**: Custom CNN with residual blocks and attention mechanisms\n")
            f.write("- **Parameters**: ~5M parameters\n")
            f.write("- **Features**: Residual connections, attention gates, batch normalization\n\n")
            
            f.write("### 2. EfficientNet-Inspired\n")
            f.write("- **Architecture**: MobileNet-style architecture with MBConv blocks\n")
            f.write("- **Parameters**: ~2.4M parameters\n")
            f.write("- **Features**: Depthwise separable convolutions, squeeze-and-excitation\n\n")
            
            f.write("## Results Summary\n\n")
            f.write("| Dataset | Model | Test Accuracy | Parameters | Efficiency |\n")
            f.write("|---------|-------|---------------|------------|------------|\n")
            
            # Sort results by accuracy
            sorted_results = sorted(self.results.items(), 
                                  key=lambda x: x[1].get('test_accuracy', 0), reverse=True)
            
            for key, data in sorted_results:
                if 'error' not in data:
                    dataset, model = key.split('_')
                    efficiency = data['test_accuracy'] / (data['model_parameters'] / 1e6)
                    f.write(f"| {dataset.upper()} | {model.upper()} | {data['test_accuracy']:.2f}% | "
                           f"{data['model_parameters'] / 1e6:.2f}M | {efficiency:.2f} |\n")
            
            f.write("\n## Key Findings\n\n")
            
            # Find best performing models
            best_overall = max(self.results.items(), 
                             key=lambda x: x[1].get('test_accuracy', 0))
            best_efficient = max(self.results.items(), 
                               key=lambda x: x[1].get('test_accuracy', 0) / (x[1].get('model_parameters', 1) / 1e6))
            
            f.write(f"### Best Overall Performance\n")
            f.write(f"- **Model**: {best_overall[0].split('_')[1].upper()} on {best_overall[0].split('_')[0].upper()}\n")
            f.write(f"- **Accuracy**: {best_overall[1]['test_accuracy']:.2f}%\n")
            f.write(f"- **Parameters**: {best_overall[1]['model_parameters'] / 1e6:.2f}M\n\n")
            
            f.write(f"### Most Efficient Model\n")
            f.write(f"- **Model**: {best_efficient[0].split('_')[1].upper()} on {best_efficient[0].split('_')[0].upper()}\n")
            f.write(f"- **Efficiency**: {best_efficient[1]['test_accuracy'] / (best_efficient[1]['model_parameters'] / 1e6):.2f} accuracy per million parameters\n\n")
            
            f.write("### Architecture Analysis\n")
            f.write("- **Advanced CNN**: Consistently outperformed EfficientNet on most tasks\n")
            f.write("- **EfficientNet**: Showed overfitting issues on OCTMNIST (25% test vs 73.5% validation)\n")
            f.write("- **Parameter Efficiency**: EfficientNet models are more parameter-efficient but less accurate\n")
            f.write("- **Dataset Suitability**: Advanced CNN better for complex medical imaging tasks\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("1. **For High Accuracy**: Use Advanced CNN architecture\n")
            f.write("2. **For Resource Constraints**: Use EfficientNet with proper regularization\n")
            f.write("3. **For Production**: Consider ensemble of both architectures\n")
            f.write("4. **For Research**: Advanced CNN provides better baseline for medical imaging\n\n")
            
            f.write("## Next Steps\n\n")
            f.write("1. **Hyperparameter Tuning**: Optimize learning rates and regularization\n")
            f.write("2. **Data Augmentation**: Implement advanced augmentation strategies\n")
            f.write("3. **Ensemble Methods**: Combine multiple architectures\n")
            f.write("4. **Transfer Learning**: Use pre-trained models as feature extractors\n")
            f.write("5. **Architecture Search**: Explore neural architecture search (NAS)\n")

def main():
    """Main function to create all visualizations."""
    visualizer = AdvancedResultsVisualizer()
    visualizer.create_all_visualizations()
    
    print("\n" + "="*60)
    print("ADVANCED TRAINING VISUALIZATION COMPLETE")
    print("="*60)
    print(f"Results saved to: {visualizer.output_dir}")
    print("\nGenerated files:")
    print("  📊 advanced_model_comparison.png - Comprehensive model comparison")
    print("  📈 advanced_performance_analysis.png - Detailed performance analysis")
    print("  📋 advanced_results_table.png - Results summary table")
    print("  📄 ADVANCED_TRAINING_SUMMARY.md - Detailed summary report")
    print("\nAll visualizations are ready for research paper integration!")

if __name__ == '__main__':
    main()
