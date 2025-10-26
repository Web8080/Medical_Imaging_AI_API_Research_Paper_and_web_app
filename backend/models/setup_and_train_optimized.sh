#!/bin/bash

echo "=========================================="
echo "Medical Imaging AI - OPTIMIZED Training"
echo "CPU-Friendly: ~24 Hours Target"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "../../venv" ]; then
    echo "Creating virtual environment..."
    cd ../..
    python3 -m venv venv
    cd backend/models
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source ../../venv/bin/activate
echo "✓ Virtual environment activated"

# Install/upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo ""
echo "Installing dependencies (this may take 5-10 minutes)..."
echo "Installing core packages..."
pip install torch torchvision numpy scikit-learn pandas matplotlib seaborn tqdm --quiet

echo "Installing medical imaging packages..."
pip install medmnist pydicom nibabel --quiet

echo "✓ All dependencies installed"

# Check GPU availability
echo ""
echo "=========================================="
echo "System Check"
echo "=========================================="
python3 << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("")
    print("🚀 GPU detected! Will be MUCH faster (<8 hours)")
else:
    print("💻 CPU mode - Optimized for ~24 hours")
print("")
EOF

echo "=========================================="
echo "Optimized Training Configuration"
echo "=========================================="
echo ""
echo "Running 6 experiments (instead of 8):"
echo ""
echo "ChestMNIST (multi-label, hardest):"
echo "  ✓ SimpleCNN - 30 epochs (~2-3h)"
echo "  ✓ AdvancedCNN - 50 epochs (~4-5h)"
echo ""
echo "DermaMNIST (7 classes):"
echo "  ✓ SimpleCNN - 30 epochs (~1-2h)"
echo "  ✓ AdvancedCNN - 50 epochs (~3-4h)"
echo ""
echo "OCTMNIST (4 classes, largest dataset):"
echo "  ✓ SimpleCNN - 30 epochs (~3-4h)"
echo "  ✓ AdvancedCNN - 50 epochs (~6-8h)"
echo ""
echo "Skipped (not needed for publication):"
echo "  ✗ EfficientNet (slower, not better than AdvancedCNN)"
echo ""
echo "Optimizations:"
echo "  • Reduced epochs (30-50 instead of 50-100)"
echo "  • Larger batch sizes (128/64 instead of 64)"
echo "  • Aggressive early stopping (patience 10)"
echo "  • Focus on best architectures only"
echo ""
echo "Total estimated time: 20-26 hours"
echo "Expected with early stopping: 18-24 hours"
echo ""
echo "Results will be saved in: training_results_extended/"
echo "Best models will be saved in: checkpoints_extended/"
echo ""
echo "Press ENTER to start optimized training, or Ctrl+C to cancel..."
read

# Start optimized training
echo ""
echo "=========================================="
echo "Starting Optimized Training"
echo "=========================================="
echo ""

./run_optimized_experiments.sh

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Check: training_results_extended/RESULTS_REPORT.md"
echo "2. Review: training_results_extended/latex_tables.tex"
echo "3. Copy LaTeX tables to research paper Section 6.3"
echo "4. Update paper with actual results"
echo ""






