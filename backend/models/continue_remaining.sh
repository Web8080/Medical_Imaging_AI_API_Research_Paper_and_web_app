#!/bin/bash

cd /Users/user/API_for_Medical_Imaging/backend/models
source ../../venv/bin/activate

echo "=========================================="
echo "Continuing Training - Experiments 4-6"
echo "=========================================="
echo ""

# Experiment 4: DermaMNIST - AdvancedCNN (50 epochs)
echo "Experiment 4/6: DermaMNIST - AdvancedCNN (50 epochs)"
echo "-------------------------------------------------------"
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
echo "Experiment 5/6: OCTMNIST - SimpleCNN (30 epochs)"
echo "--------------------------------------------------"
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
echo "Experiment 6/6: OCTMNIST - AdvancedCNN (50 epochs)"
echo "-----------------------------------------------------"
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
echo "All experiments completed!"
echo "=========================================="
echo ""

# Generate analysis
echo "Generating analysis report..."
python3 analyze_extended_results.py

echo ""
echo "=========================================="
echo "✓ Training and Analysis Complete!"
echo "=========================================="
echo ""



