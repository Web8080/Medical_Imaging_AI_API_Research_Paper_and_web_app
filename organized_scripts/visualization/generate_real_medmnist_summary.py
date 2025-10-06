#!/usr/bin/env python3
"""
Generate comprehensive summary and visualizations for real MedMNIST training results.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_real_medmnist_results():
    """Load real MedMNIST training results."""
    results_dir = Path('results/real_medmnist_training/metrics')
    
    results = {}
    
    # Load individual dataset results
    for dataset in ['chestmnist', 'octmnist']:
        history_file = results_dir / f'{dataset}_history.json'
        if history_file.exists():
            with open(history_file, 'r') as f:
                results[dataset] = json.load(f)
    
    # Load training summary
    summary_file = results_dir / 'training_summary.json'
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            results['summary'] = json.load(f)
    
    return results

def create_real_medmnist_plots(results):
    """Create visualization plots for real MedMNIST training."""
    
    # Create output directory
    output_dir = Path('results/real_medmnist_training/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Training History Comparison
    plt.figure(figsize=(15, 10))
    
    datasets = ['chestmnist', 'octmnist']
    colors = ['#2E86AB', '#A23B72']
    
    for i, dataset in enumerate(datasets):
        if dataset in results:
            history = results[dataset]
            epochs = range(1, len(history['train_losses']) + 1)
            
            plt.subplot(2, 2, 1)
            plt.plot(epochs, history['train_losses'], color=colors[i], label=f'{dataset.upper()} Train', linewidth=2)
            plt.plot(epochs, history['val_losses'], color=colors[i], linestyle='--', label=f'{dataset.upper()} Val', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 2)
            plt.plot(epochs, history['train_accs'], color=colors[i], label=f'{dataset.upper()} Train', linewidth=2)
            plt.plot(epochs, history['val_accs'], color=colors[i], linestyle='--', label=f'{dataset.upper()} Val', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.title('Training and Validation Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    # 2. Final Performance Comparison
    if 'summary' in results:
        summary = results['summary']
        
        plt.subplot(2, 2, 3)
        datasets = list(summary.keys())
        test_accuracies = [summary[dataset]['test_accuracy'] for dataset in datasets]
        
        bars = plt.bar(datasets, test_accuracies, color=colors[:len(datasets)], alpha=0.7)
        plt.xlabel('Dataset')
        plt.ylabel('Test Accuracy (%)')
        plt.title('Final Test Performance')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, acc in zip(bars, test_accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 3. Training Time Comparison
        plt.subplot(2, 2, 4)
        training_times = [summary[dataset]['training_time'] for dataset in datasets]
        
        bars = plt.bar(datasets, training_times, color=colors[:len(datasets)], alpha=0.7)
        plt.xlabel('Dataset')
        plt.ylabel('Training Time (seconds)')
        plt.title('Training Time Comparison')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, time in zip(bars, training_times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{time:.0f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'real_medmnist_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Performance Metrics Heatmap
    if 'summary' in results:
        plt.figure(figsize=(10, 6))
        
        summary = results['summary']
        datasets = list(summary.keys())
        metrics = ['test_accuracy', 'test_loss', 'training_time']
        
        # Create matrix
        matrix = []
        for dataset in datasets:
            row = [
                summary[dataset]['test_accuracy'],
                summary[dataset]['test_loss'],
                summary[dataset]['training_time'] / 100  # Normalize for visualization
            ]
            matrix.append(row)
        
        matrix = np.array(matrix)
        
        # Create heatmap
        sns.heatmap(matrix, 
                    xticklabels=['Test Accuracy (%)', 'Test Loss', 'Training Time (×100s)'],
                    yticklabels=[d.upper() for d in datasets],
                    annot=True, 
                    fmt='.2f',
                    cmap='RdYlBu_r',
                    cbar_kws={'label': 'Value'})
        
        plt.title('Real MedMNIST Training - Performance Metrics Heatmap')
        plt.tight_layout()
        plt.savefig(output_dir / 'real_medmnist_metrics_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Training Progress Detailed
    plt.figure(figsize=(16, 8))
    
    for i, dataset in enumerate(datasets):
        if dataset in results:
            history = results[dataset]
            epochs = range(1, len(history['train_losses']) + 1)
            
            plt.subplot(2, 3, i*3 + 1)
            plt.plot(epochs, history['train_losses'], 'b-', label='Train', linewidth=2)
            plt.plot(epochs, history['val_losses'], 'r-', label='Validation', linewidth=2)
            plt.title(f'{dataset.upper()} - Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 3, i*3 + 2)
            plt.plot(epochs, history['train_accs'], 'b-', label='Train', linewidth=2)
            plt.plot(epochs, history['val_accs'], 'r-', label='Validation', linewidth=2)
            plt.title(f'{dataset.upper()} - Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 3, i*3 + 3)
            # Learning rate if available
            if 'lr' in history:
                lr_values = [history['lr']] * len(epochs)
                plt.plot(epochs, lr_values, 'g-', linewidth=2)
                plt.title(f'{dataset.upper()} - Learning Rate')
                plt.xlabel('Epoch')
                plt.ylabel('Learning Rate')
                plt.grid(True, alpha=0.3)
            else:
                # Show convergence
                train_acc = history['train_accs']
                val_acc = history['val_accs']
                convergence = [abs(t - v) for t, v in zip(train_acc, val_acc)]
                plt.plot(epochs, convergence, 'purple', linewidth=2)
                plt.title(f'{dataset.upper()} - Train-Val Gap')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy Difference (%)')
                plt.grid(True, alpha=0.3)
    
    plt.suptitle('Real MedMNIST Training - Detailed Progress Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'real_medmnist_detailed_progress.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Summary Statistics
    plt.figure(figsize=(12, 8))
    
    if 'summary' in results:
        summary = results['summary']
        
        # Create summary statistics
        datasets = list(summary.keys())
        accuracies = [summary[dataset]['test_accuracy'] for dataset in datasets]
        losses = [summary[dataset]['test_loss'] for dataset in datasets]
        times = [summary[dataset]['training_time'] for dataset in datasets]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Accuracy comparison
        axes[0, 0].bar(datasets, accuracies, color=['#2E86AB', '#A23B72'], alpha=0.7)
        axes[0, 0].set_title('Test Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].set_xticklabels([d.upper() for d in datasets])
        for i, acc in enumerate(accuracies):
            axes[0, 0].text(i, acc + 1, f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Loss comparison
        axes[0, 1].bar(datasets, losses, color=['#2E86AB', '#A23B72'], alpha=0.7)
        axes[0, 1].set_title('Test Loss Comparison')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_xticklabels([d.upper() for d in datasets])
        for i, loss in enumerate(losses):
            axes[0, 1].text(i, loss + 0.01, f'{loss:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Training time comparison
        axes[1, 0].bar(datasets, times, color=['#2E86AB', '#A23B72'], alpha=0.7)
        axes[1, 0].set_title('Training Time Comparison')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].set_xticklabels([d.upper() for d in datasets])
        for i, time in enumerate(times):
            axes[1, 0].text(i, time + 5, f'{time:.0f}s', ha='center', va='bottom', fontweight='bold')
        
        # Performance summary
        axes[1, 1].text(0.5, 0.7, f'Best Performance: {max(accuracies):.1f}%', 
                       ha='center', va='center', fontsize=14, fontweight='bold')
        axes[1, 1].text(0.5, 0.5, f'Average Accuracy: {np.mean(accuracies):.1f}%', 
                       ha='center', va='center', fontsize=12)
        axes[1, 1].text(0.5, 0.3, f'Total Training Time: {sum(times):.0f}s', 
                       ha='center', va='center', fontsize=12)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.suptitle('Real MedMNIST Training - Summary Statistics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'real_medmnist_summary_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Generated real MedMNIST plots in {output_dir}")
    return output_dir

def main():
    """Generate all real MedMNIST training plots."""
    logger.info("Generating real MedMNIST training plots...")
    
    # Load results
    results = load_real_medmnist_results()
    if not results:
        logger.error("No results found")
        return
    
    # Create plots
    output_dir = create_real_medmnist_plots(results)
    
    # Copy plots to research paper visualizations folder
    research_viz_dir = Path('training_results/research_paper_visualizations')
    research_viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy all generated plots
    for plot_file in output_dir.glob('*.png'):
        import shutil
        shutil.copy2(plot_file, research_viz_dir / plot_file.name)
        logger.info(f"Copied {plot_file.name} to research paper visualizations")
    
    print("\n" + "="*60)
    print("REAL MEDMNIST TRAINING PLOTS GENERATED")
    print("="*60)
    print(f"Plots saved to: {output_dir}")
    print(f"Copied to: {research_viz_dir}")
    print("Generated plots:")
    for plot_file in output_dir.glob('*.png'):
        print(f"  - {plot_file.name}")
    print("="*60)

if __name__ == '__main__':
    main()
