#!/bin/bash

# Optimized Training for CPU (~24 hours or less)
# Focuses on most important experiments with reduced epochs but still scientifically valid

set -e  # Exit on error

echo "=========================================="
echo "OPTIMIZED Extended Training (CPU-Friendly)"
echo "Target Time: ~24 hours on CPU"
echo "=========================================="
echo ""

# Create results directory
mkdir -p training_results_extended
mkdir -p checkpoints_extended

# Experiment 1: ChestMNIST - SimpleCNN (30 epochs)
# Multi-label is hardest, so use SimpleCNN baseline
echo "Experiment 1/6: ChestMNIST - SimpleCNN (30 epochs)"
echo "--------------------------------------------------"
echo "Estimated time: ~2-3 hours"
python3 train_extended_epochs.py \
    --dataset chestmnist \
    --model simple \
    --epochs 30 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --weight_decay 0.0001 \
    --scheduler plateau \
    --early_stopping \
    --early_stopping_patience 10 \
    --save_frequency 15 \
    --experiment_name chestmnist_simple_30epochs_optimized
echo "✓ Completed 1/6"
echo ""

# Experiment 2: ChestMNIST - AdvancedCNN (50 epochs)
# Best architecture for this difficult task
echo "Experiment 2/6: ChestMNIST - AdvancedCNN (50 epochs)"
echo "------------------------------------------------------"
echo "Estimated time: ~4-5 hours"
python3 train_extended_epochs.py \
    --dataset chestmnist \
    --model advanced \
    --epochs 50 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --weight_decay 0.0001 \
    --scheduler cosine \
    --early_stopping \
    --early_stopping_patience 10 \
    --save_frequency 15 \
    --experiment_name chestmnist_advanced_50epochs_optimized
echo "✓ Completed 2/6"
echo ""

# Experiment 3: DermaMNIST - SimpleCNN (30 epochs)
# Baseline for dermatology
echo "Experiment 3/6: DermaMNIST - SimpleCNN (30 epochs)"
echo "----------------------------------------------------"
echo "Estimated time: ~1-2 hours"
python3 train_extended_epochs.py \
    --dataset dermamnist \
    --model simple \
    --epochs 30 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --weight_decay 0.0001 \
    --scheduler plateau \
    --early_stopping \
    --early_stopping_patience 10 \
    --save_frequency 15 \
    --experiment_name dermamnist_simple_30epochs_optimized
echo "✓ Completed 3/6"
echo ""

# Experiment 4: DermaMNIST - AdvancedCNN (50 epochs)
# Best architecture for publication
echo "Experiment 4/6: DermaMNIST - AdvancedCNN (50 epochs)"
echo "-------------------------------------------------------"
echo "Estimated time: ~3-4 hours"
python3 train_extended_epochs.py \
    --dataset dermamnist \
    --model advanced \
    --epochs 50 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --weight_decay 0.0001 \
    --scheduler cosine \
    --early_stopping \
    --early_stopping_patience 10 \
    --save_frequency 15 \
    --experiment_name dermamnist_advanced_50epochs_optimized
echo "✓ Completed 4/6"
echo ""

# Experiment 5: OCTMNIST - SimpleCNN (30 epochs)
# Baseline for OCT
echo "Experiment 5/6: OCTMNIST - SimpleCNN (30 epochs)"
echo "--------------------------------------------------"
echo "Estimated time: ~3-4 hours"
python3 train_extended_epochs.py \
    --dataset octmnist \
    --model simple \
    --epochs 30 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --weight_decay 0.0001 \
    --scheduler plateau \
    --early_stopping \
    --early_stopping_patience 10 \
    --save_frequency 15 \
    --experiment_name octmnist_simple_30epochs_optimized
echo "✓ Completed 5/6"
echo ""

# Experiment 6: OCTMNIST - AdvancedCNN (50 epochs)
# Best results expected here (largest dataset)
echo "Experiment 6/6: OCTMNIST - AdvancedCNN (50 epochs)"
echo "-----------------------------------------------------"
echo "Estimated time: ~6-8 hours"
python3 train_extended_epochs.py \
    --dataset octmnist \
    --model advanced \
    --epochs 50 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --weight_decay 0.0001 \
    --scheduler cosine \
    --early_stopping \
    --early_stopping_patience 10 \
    --save_frequency 15 \
    --experiment_name octmnist_advanced_50epochs_optimized
echo "✓ Completed 6/6"
echo ""

echo "=========================================="
echo "All optimized experiments completed!"
echo "=========================================="
echo ""
echo "Total experiments: 6 (reduced from 8)"
echo "Total time: ~20-26 hours (target: <24 hours with early stopping)"
echo ""
echo "Skipped experiments (not critical for publication):"
echo "  - EfficientNet on DermaMNIST (good but AdvancedCNN is better)"
echo "  - EfficientNet on OCTMNIST (performs poorly on grayscale)"
echo ""
echo "Results saved in: training_results_extended/"
echo "Checkpoints saved in: checkpoints_extended/"
echo ""

# Generate summary report
echo "Generating analysis report..."
python3 analyze_extended_results.py

echo ""
echo "=========================================="
echo "✓ Training and Analysis Complete!"
echo "=========================================="
echo ""
echo "Check results: training_results_extended/RESULTS_REPORT.md"
echo "LaTeX tables: training_results_extended/latex_tables.tex"
echo ""



