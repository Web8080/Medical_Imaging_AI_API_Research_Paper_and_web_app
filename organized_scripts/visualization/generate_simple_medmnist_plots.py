#!/usr/bin/env python3
"""
Generate simple visualization plots for MedMNIST training results.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_medmnist_results():
    """Load MedMNIST training results."""
    results_dir = Path('results/real_medmnist_training/metrics')
    
    results = {}
    
    # Load individual dataset results
    for dataset in ['chestmnist', 'octmnist']:
        history_file = results_dir / f'{dataset}_history.json'
        if history_file.exists():
            with open(history_file, 'r') as f:
                results[dataset] = json.load(f)
    
    return results

def create_medmnist_plots(results):
    """Create visualization plots for MedMNIST training."""
    
    # Create output directory
    output_dir = Path('results/real_medmnist_training/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    datasets = list(results.keys())
    if not datasets:
        logger.error("No datasets found")
        return output_dir
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # 1. Training Progress Comparison
    plt.figure(figsize=(15, 10))
    
    for i, dataset in enumerate(datasets):
        history = results[dataset]
        epochs = range(1, len(history['train_losses']) + 1)
        
        plt.subplot(2, 2, 1)
        plt.plot(epochs, history['train_losses'], color=colors[i], label=f'{dataset.upper()} Train', linewidth=2, marker='o')
        plt.plot(epochs, history['val_losses'], color=colors[i], linestyle='--', label=f'{dataset.upper()} Val', linewidth=2, marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(epochs, history['train_accs'], color=colors[i], label=f'{dataset.upper()} Train', linewidth=2, marker='o')
        plt.plot(epochs, history['val_accs'], color=colors[i], linestyle='--', label=f'{dataset.upper()} Val', linewidth=2, marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 2. Final Performance Summary
    plt.subplot(2, 2, 3)
    final_accuracies = []
    final_losses = []
    dataset_names = []
    
    for dataset in datasets:
        history = results[dataset]
        final_acc = history['val_accs'][-1]  # Last validation accuracy
        final_loss = history['val_losses'][-1]  # Last validation loss
        final_accuracies.append(final_acc)
        final_losses.append(final_loss)
        dataset_names.append(dataset.upper())
    
    bars = plt.bar(dataset_names, final_accuracies, color=colors[:len(datasets)], alpha=0.7)
    plt.xlabel('Dataset')
    plt.ylabel('Final Validation Accuracy (%)')
    plt.title('Final Performance Comparison')
    
    # Add value labels
    for bar, acc in zip(bars, final_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 3. Training Configuration
    plt.subplot(2, 2, 4)
    config_data = []
    config_labels = []
    
    for dataset in datasets:
        history = results[dataset]
        config_data.append([
            history.get('epochs', 0),
            history.get('batch_size', 0),
            history.get('lr', 0) * 1000  # Convert to readable format
        ])
        config_labels.append(dataset.upper())
    
    # Create a simple configuration display
    plt.text(0.1, 0.8, 'Training Configuration:', fontsize=14, fontweight='bold')
    y_pos = 0.7
    for i, (dataset, config) in enumerate(zip(config_labels, config_data)):
        plt.text(0.1, y_pos, f'{dataset}:', fontsize=12, fontweight='bold')
        plt.text(0.2, y_pos-0.05, f'Epochs: {config[0]}, Batch: {config[1]}, LR: {config[2]:.3f}', fontsize=10)
        y_pos -= 0.15
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('Training Configuration')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'medmnist_training_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Individual Dataset Analysis
    for i, dataset in enumerate(datasets):
        history = results[dataset]
        epochs = range(1, len(history['train_losses']) + 1)
        
        plt.figure(figsize=(12, 8))
        
        # Loss progression
        plt.subplot(2, 2, 1)
        plt.plot(epochs, history['train_losses'], 'b-', label='Training Loss', linewidth=2, marker='o')
        plt.plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2, marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{dataset.upper()} - Loss Progression')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy progression
        plt.subplot(2, 2, 2)
        plt.plot(epochs, history['train_accs'], 'b-', label='Training Accuracy', linewidth=2, marker='o')
        plt.plot(epochs, history['val_accs'], 'r-', label='Validation Accuracy', linewidth=2, marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title(f'{dataset.upper()} - Accuracy Progression')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Convergence analysis
        plt.subplot(2, 2, 3)
        train_val_gap = [abs(t - v) for t, v in zip(history['train_accs'], history['val_accs'])]
        plt.plot(epochs, train_val_gap, 'purple', linewidth=2, marker='d')
        plt.xlabel('Epoch')
        plt.ylabel('Train-Val Accuracy Gap (%)')
        plt.title(f'{dataset.upper()} - Convergence Analysis')
        plt.grid(True, alpha=0.3)
        
        # Performance summary
        plt.subplot(2, 2, 4)
        final_train_acc = history['train_accs'][-1]
        final_val_acc = history['val_accs'][-1]
        final_train_loss = history['train_losses'][-1]
        final_val_loss = history['val_losses'][-1]
        
        summary_text = f"""
        Final Performance:
        
        Training Accuracy: {final_train_acc:.2f}%
        Validation Accuracy: {final_val_acc:.2f}%
        
        Training Loss: {final_train_loss:.4f}
        Validation Loss: {final_val_loss:.4f}
        
        Configuration:
        Epochs: {history.get('epochs', 'N/A')}
        Batch Size: {history.get('batch_size', 'N/A')}
        Learning Rate: {history.get('lr', 'N/A')}
        """
        
        plt.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title(f'{dataset.upper()} - Performance Summary')
        
        plt.suptitle(f'{dataset.upper()} Training Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'{dataset}_detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Comparative Analysis
    plt.figure(figsize=(12, 8))
    
    # Performance comparison
    plt.subplot(2, 2, 1)
    bars = plt.bar(dataset_names, final_accuracies, color=colors[:len(datasets)], alpha=0.7)
    plt.xlabel('Dataset')
    plt.ylabel('Final Validation Accuracy (%)')
    plt.title('Final Accuracy Comparison')
    for bar, acc in zip(bars, final_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Loss comparison
    plt.subplot(2, 2, 2)
    bars = plt.bar(dataset_names, final_losses, color=colors[:len(datasets)], alpha=0.7)
    plt.xlabel('Dataset')
    plt.ylabel('Final Validation Loss')
    plt.title('Final Loss Comparison')
    for bar, loss in zip(bars, final_losses):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{loss:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Training efficiency
    plt.subplot(2, 2, 3)
    epochs_trained = [results[dataset].get('epochs', 0) for dataset in datasets]
    efficiency = [acc/epoch for acc, epoch in zip(final_accuracies, epochs_trained)]
    bars = plt.bar(dataset_names, efficiency, color=colors[:len(datasets)], alpha=0.7)
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy per Epoch')
    plt.title('Training Efficiency')
    for bar, eff in zip(bars, efficiency):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{eff:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Best performance highlight
    plt.subplot(2, 2, 4)
    best_idx = np.argmax(final_accuracies)
    best_dataset = dataset_names[best_idx]
    best_acc = final_accuracies[best_idx]
    
    plt.text(0.5, 0.5, f'Best Performance:\n{best_dataset}\n{best_acc:.1f}%', 
             ha='center', va='center', fontsize=16, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('Best Performance')
    
    plt.suptitle('MedMNIST Training - Comparative Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'medmnist_comparative_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Generated MedMNIST plots in {output_dir}")
    return output_dir

def main():
    """Generate all MedMNIST training plots."""
    logger.info("Generating MedMNIST training plots...")
    
    # Load results
    results = load_medmnist_results()
    if not results:
        logger.error("No results found")
        return
    
    # Create plots
    output_dir = create_medmnist_plots(results)
    
    # Copy plots to research paper visualizations folder
    research_viz_dir = Path('training_results/research_paper_visualizations')
    research_viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy all generated plots
    for plot_file in output_dir.glob('*.png'):
        import shutil
        shutil.copy2(plot_file, research_viz_dir / plot_file.name)
        logger.info(f"Copied {plot_file.name} to research paper visualizations")
    
    print("\n" + "="*60)
    print("MEDMNIST TRAINING PLOTS GENERATED")
    print("="*60)
    print(f"Plots saved to: {output_dir}")
    print(f"Copied to: {research_viz_dir}")
    print("Generated plots:")
    for plot_file in output_dir.glob('*.png'):
        print(f"  - {plot_file.name}")
    print("="*60)

if __name__ == '__main__':
    main()
