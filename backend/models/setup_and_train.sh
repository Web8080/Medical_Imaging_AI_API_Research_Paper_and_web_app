#!/bin/bash

echo "=========================================="
echo "Medical Imaging AI - Extended Training Setup"
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
    print("🚀 GPU detected! Training will be FAST (8-12 hours)")
else:
    print("⚠️  No GPU detected. Training will use CPU (24-48 hours)")
print("")
EOF

echo "=========================================="
echo "Ready to Start Training"
echo "=========================================="
echo ""
echo "The training will run 8 experiments:"
echo "  • ChestMNIST: SimpleCNN (50 epochs) + AdvancedCNN (100 epochs)"
echo "  • DermaMNIST: SimpleCNN (50 epochs) + AdvancedCNN (100 epochs) + EfficientNet (100 epochs)"
echo "  • OCTMNIST: SimpleCNN (50 epochs) + AdvancedCNN (100 epochs) + EfficientNet (100 epochs)"
echo ""
echo "Results will be saved in: training_results_extended/"
echo "Best models will be saved in: checkpoints_extended/"
echo ""
echo "Press ENTER to start training, or Ctrl+C to cancel..."
read

# Start training
echo ""
echo "=========================================="
echo "Starting Extended Training"
echo "=========================================="
echo ""

./run_all_extended_experiments.sh

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run: python3 analyze_extended_results.py"
echo "2. Check: training_results_extended/RESULTS_REPORT.md"
echo "3. Copy LaTeX tables to research paper"
echo ""



