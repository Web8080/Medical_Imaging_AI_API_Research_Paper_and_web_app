#!/bin/bash

# Extended Training Experiments for Research Paper
# This script runs all experiments with 50-100 epochs

set -e  # Exit on error

echo "=========================================="
echo "Starting Extended Training Experiments"
echo "=========================================="
echo ""

# Create results directory
mkdir -p training_results_extended
mkdir -p checkpoints_extended

# Experiment 1: ChestMNIST - Simple CNN (50 epochs)
echo "Experiment 1/8: ChestMNIST - Simple CNN (50 epochs)"
echo "--------------------------------------------------"
python3 train_extended_epochs.py \
    --dataset chestmnist \
    --model simple \
    --epochs 50 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --weight_decay 0.0001 \
    --scheduler plateau \
    --early_stopping \
    --early_stopping_patience 15 \
    --experiment_name chestmnist_simple_50epochs
echo ""

# Experiment 2: ChestMNIST - Advanced CNN (100 epochs)
echo "Experiment 2/8: ChestMNIST - Advanced CNN (100 epochs)"
echo "------------------------------------------------------"
python3 train_extended_epochs.py \
    --dataset chestmnist \
    --model advanced \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --weight_decay 0.0001 \
    --scheduler cosine \
    --early_stopping \
    --early_stopping_patience 20 \
    --experiment_name chestmnist_advanced_100epochs
echo ""

# Experiment 3: DermaMNIST - Simple CNN (50 epochs)
echo "Experiment 3/8: DermaMNIST - Simple CNN (50 epochs)"
echo "----------------------------------------------------"
python3 train_extended_epochs.py \
    --dataset dermamnist \
    --model simple \
    --epochs 50 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --weight_decay 0.0001 \
    --scheduler plateau \
    --early_stopping \
    --early_stopping_patience 15 \
    --experiment_name dermamnist_simple_50epochs
echo ""

# Experiment 4: DermaMNIST - Advanced CNN (100 epochs)
echo "Experiment 4/8: DermaMNIST - Advanced CNN (100 epochs)"
echo "-------------------------------------------------------"
python3 train_extended_epochs.py \
    --dataset dermamnist \
    --model advanced \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --weight_decay 0.0001 \
    --scheduler cosine \
    --early_stopping \
    --early_stopping_patience 20 \
    --experiment_name dermamnist_advanced_100epochs
echo ""

# Experiment 5: DermaMNIST - EfficientNet (100 epochs)
echo "Experiment 5/8: DermaMNIST - EfficientNet (100 epochs)"
echo "-------------------------------------------------------"
python3 train_extended_epochs.py \
    --dataset dermamnist \
    --model efficientnet \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --weight_decay 0.0001 \
    --scheduler cosine \
    --early_stopping \
    --early_stopping_patience 20 \
    --experiment_name dermamnist_efficientnet_100epochs
echo ""

# Experiment 6: OCTMNIST - Simple CNN (50 epochs)
echo "Experiment 6/8: OCTMNIST - Simple CNN (50 epochs)"
echo "--------------------------------------------------"
python3 train_extended_epochs.py \
    --dataset octmnist \
    --model simple \
    --epochs 50 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --weight_decay 0.0001 \
    --scheduler plateau \
    --early_stopping \
    --early_stopping_patience 15 \
    --experiment_name octmnist_simple_50epochs
echo ""

# Experiment 7: OCTMNIST - Advanced CNN (100 epochs)
echo "Experiment 7/8: OCTMNIST - Advanced CNN (100 epochs)"
echo "-----------------------------------------------------"
python3 train_extended_epochs.py \
    --dataset octmnist \
    --model advanced \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --weight_decay 0.0001 \
    --scheduler cosine \
    --early_stopping \
    --early_stopping_patience 20 \
    --experiment_name octmnist_advanced_100epochs
echo ""

# Experiment 8: OCTMNIST - EfficientNet (100 epochs)
echo "Experiment 8/8: OCTMNIST - EfficientNet (100 epochs)"
echo "-----------------------------------------------------"
python3 train_extended_epochs.py \
    --dataset octmnist \
    --model efficientnet \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --weight_decay 0.0001 \
    --scheduler cosine \
    --early_stopping \
    --early_stopping_patience 20 \
    --experiment_name octmnist_efficientnet_100epochs
echo ""

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Results saved in: training_results_extended/"
echo "Checkpoints saved in: checkpoints_extended/"
echo ""

# Generate summary report
python3 analyze_extended_results.py



