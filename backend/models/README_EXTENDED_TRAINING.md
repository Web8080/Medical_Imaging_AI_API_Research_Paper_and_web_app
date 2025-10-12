# Extended Training for Research Paper (50-100 Epochs)

This directory contains scripts for conducting comprehensive training experiments that generate publication-quality results for the research paper.

## Overview

The extended training experiments address the scientific integrity concerns identified in the paper audit by:

1. **Proper Training Duration**: 50-100 epochs instead of 3 epochs
2. **Comprehensive Validation**: Separate validation sets with early stopping
3. **Multiple Architectures**: SimpleCNN, AdvancedCNN, EfficientNet-Inspired
4. **Complete Metrics**: Accuracy, precision, recall, F1-score, confusion matrices
5. **Training Analysis**: Learning curves, convergence plots, overfitting analysis

## Quick Start

### Run All Experiments

To run all 8 experiments (approximately 8-12 hours on GPU, 24-48 hours on CPU):

```bash
cd /Users/user/API_for_Medical_Imaging/backend/models
chmod +x run_all_extended_experiments.sh
./run_all_extended_experiments.sh
```

###Run Individual Experiments

```bash
# Example: Train AdvancedCNN on DermaMNIST for 100 epochs
python3 train_extended_epochs.py \
    --dataset dermamnist \
    --model advanced \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --scheduler cosine \
    --early_stopping \
    --early_stopping_patience 20 \
    --experiment_name dermamnist_advanced_100epochs
```

## Experiment Configuration

### Datasets
- **ChestMNIST**: 112,120 chest X-rays, 14-class multi-label classification
- **DermaMNIST**: 10,015 dermatoscopy images, 7-class single-label classification
- **OCTMNIST**: 109,309 retinal OCT scans, 4-class single-label classification

### Model Architectures

1. **SimpleCNN** (~1.1M parameters)
   - 3 convolutional layers with pooling
   - Global average pooling
   - Dropout regularization
   - Baseline architecture

2. **AdvancedCNN** (~5M parameters)
   - Residual blocks with skip connections
   - Squeeze-and-Excitation attention mechanisms
   - Batch normalization
   - State-of-the-art design

3. **EfficientNet-Inspired** (~2.4M parameters)
   - Mobile inverted bottleneck convolution (MBConv) blocks
   - Depthwise separable convolutions
   - SE attention
   - Parameter-efficient architecture

### Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | AdamW | With weight decay 1e-4 |
| Learning Rate | 0.001 (0.0005 for EfficientNet) | Initial LR |
| Scheduler | ReduceLROnPlateau or CosineAnnealingLR | Adaptive LR adjustment |
| Early Stopping | Enabled | Patience 15-20 epochs |
| Batch Size | 64 | Fits in 8GB GPU memory |
| Epochs | 50-100 | Configurable per experiment |
| Data Augmentation | Normalization, rotation, flipping | Medical imaging appropriate |

## Results and Analysis

### Generated Files

After training, the following files will be created:

```
training_results_extended/
├── results_chestmnist_simple_50epochs.json
├── results_dermamnist_advanced_100epochs.json
├── training_history_chestmnist_simple_50epochs.png
├── confusion_matrix_dermamnist_advanced_100epochs.png
├── summary_table.csv
├── RESULTS_REPORT.md
├── latex_tables.tex
└── research_paper_summary.json

checkpoints_extended/
├── best_model.pth (for each experiment)
└── checkpoint_epoch_*.pth
```

### Analyze Results

After training completes, analyze all results:

```bash
python3 analyze_extended_results.py
```

This generates:
- Summary table (CSV and Markdown)
- Comparison plots across datasets and models
- LaTeX tables for research paper
- Performance heatmaps
- Research paper summary JSON

## Integration with Research Paper

### Step 1: Run Training

```bash
./run_all_extended_experiments.sh
```

### Step 2: Analyze Results

```bash
python3 analyze_extended_results.py
```

### Step 3: Update Paper

1. Open `training_results_extended/latex_tables.tex`
2. Copy the generated LaTeX tables
3. Paste into research paper Section 6 (Results)
4. Replace placeholder text about "Extended Training Results" with actual results
5. Copy performance metrics into Discussion section

### Step 4: Add Figures

Copy generated figures to paper directory:

```bash
cp training_results_extended/comparison_*.png ../../docs/research_paper/figures/
cp training_results_extended/heatmap_*.png ../../docs/research_paper/figures/
```

Update LaTeX to include figures:

```latex
\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/comparison_test_accuracy.png}
\caption{Test Accuracy Comparison Across Datasets and Models}
\label{fig:comparison_accuracy}
\end{figure}
```

## Expected Results

Based on proper training, we anticipate:

| Dataset | Model | Expected Accuracy | Training Time (GPU) |
|---------|-------|-------------------|---------------------|
| ChestMNIST | SimpleCNN | 60-65% | ~2 hours |
| ChestMNIST | AdvancedCNN | 65-70% | ~4 hours |
| DermaMNIST | SimpleCNN | 75-80% | ~1 hour |
| DermaMNIST | AdvancedCNN | 80-85% | ~2 hours |
| DermaMNIST | EfficientNet | 78-83% | ~2 hours |
| OCTMNIST | SimpleCNN | 80-85% | ~3 hours |
| OCTMNIST | AdvancedCNN | 85-90% | ~5 hours |
| OCTMNIST | EfficientNet | 30-40% | ~5 hours (poor for grayscale) |

**Note**: ChestMNIST is multi-label classification (14 classes), which is inherently more difficult. OCTMNIST has the largest training set (109K images), enabling best performance.

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
python3 train_extended_epochs.py --batch_size 32 ...
```

### Training Too Slow on CPU

Use fewer epochs or enable early stopping:
```bash
python3 train_extended_epochs.py --epochs 50 --early_stopping ...
```

### Disk Space Issues

Results and checkpoints require ~5-10GB. Clean old checkpoints:
```bash
rm -rf checkpoints_extended/checkpoint_epoch_*.pth  # Keep only best_model.pth
```

## Hardware Requirements

### Minimum
- CPU: Multi-core (4+ cores recommended)
- RAM: 16GB
- Disk: 10GB free space
- Training Time: 24-48 hours

### Recommended
- GPU: NVIDIA with 8GB+ VRAM (RTX 2060, GTX 1080, or better)
- RAM: 32GB
- Disk: 20GB free space
- Training Time: 8-12 hours

## Scientific Integrity

This extended training addresses the following concerns from the paper audit:

1. ✅ **Proper Training Duration**: 50-100 epochs ensure model convergence
2. ✅ **Validation Strategy**: Separate validation set with early stopping prevents overfitting
3. ✅ **Comprehensive Metrics**: Precision, recall, F1-score, not just accuracy
4. ✅ **Reproducibility**: Fixed random seeds, saved configurations, checkpoint management
5. ✅ **Transparency**: All hyperparameters documented, training curves saved
6. ✅ **Statistical Rigor**: Multiple architectures, multiple datasets, error analysis

## Citation

If you use this training framework, please cite:

```bibtex
@article{medical_imaging_ai_api_2024,
  title={A Scalable API Framework for Medical Imaging AI},
  author={Medical Imaging AI Research Team},
  year={2024}
}
```

## License

This code is provided for research purposes. See LICENSE file for details.

## Contact

For questions about the extended training experiments:
- Open an issue in the GitHub repository
- Check training logs in `training_extended.log`
- Review results in `training_results_extended/RESULTS_REPORT.md`



