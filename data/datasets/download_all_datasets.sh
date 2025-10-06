#!/bin/bash
# Master script to download all medical imaging datasets

echo "Medical Imaging AI - Dataset Download Script"
echo "============================================="

# Create data directory
mkdir -p data/datasets

echo "Setting up dataset download instructions..."

# Download chest X-ray dataset
echo "1. Chest X-ray Dataset:"
echo "   Visit: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia"
echo "   Download and extract to: data/datasets/chest_xray/"

# Download skin cancer dataset
echo "2. Skin Cancer Dataset:"
echo "   Visit: https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign"
echo "   Download and extract to: data/datasets/skin_cancer/"

# Download retinal disease dataset
echo "3. Retinal Disease Dataset:"
echo "   Visit: https://www.kaggle.com/datasets/paultimothymooney/retinal-disease-classification"
echo "   Download and extract to: data/datasets/retinal_disease/"

# Download Medical MNIST dataset
echo "4. Medical MNIST Dataset:"
echo "   Visit: https://www.kaggle.com/datasets/andrewmvd/medical-mnist"
echo "   Download and extract to: data/datasets/medical_mnist/"

# Download BRATS dataset
echo "5. BRATS 2021 Dataset:"
echo "   Visit: https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation"
echo "   Download and extract to: data/datasets/brats2021/"

# Download LIDC-IDRI dataset
echo "6. LIDC-IDRI Dataset:"
echo "   Visit: https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images"
echo "   Download and extract to: data/datasets/lidc_idri/"

echo ""
echo "After downloading datasets, run:"
echo "python scripts/preprocess_datasets.py"
echo "python scripts/train_models.py --dataset chest_xray --model attention_unet"
