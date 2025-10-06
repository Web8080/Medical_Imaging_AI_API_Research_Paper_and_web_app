#!/usr/bin/env python3
"""
Complete Pipeline Test Script
Tests the entire Medical Imaging AI pipeline with real downloaded images
"""

import os
import sys
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project paths
sys.path.append('src')
sys.path.append('api')

def load_trained_model(model_path, model_type='simple_cnn'):
    """Load a trained model"""
    try:
        if 'chestmnist' in model_path.lower():
            # Load simple CNN model for ChestMNIST
            from src.models.simple_cnn import SimpleCNN
            model = SimpleCNN(num_classes=14, input_channels=1)  # ChestMNIST has 14 classes
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            return model
        elif 'research' in model_path.lower():
            # Load Research Paper model
            from src.models.research_paper_cnn import ResearchPaperCNN
            model = ResearchPaperCNN(num_classes=14, input_channels=1)
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            return model
        elif 'advanced' in model_path.lower():
            # Load Advanced CNN model
            from src.models.advanced_cnn import AdvancedCNN
            model = AdvancedCNN(num_classes=7, input_channels=3)  # DermaMNIST has 7 classes, RGB
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            return model
        elif 'efficientnet' in model_path.lower():
            # Load EfficientNet model
            from src.models.efficientnet import EfficientNetModel
            model = EfficientNetModel(num_classes=7, input_channels=3)  # DermaMNIST has 7 classes, RGB
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_image(image_path, target_size=(28, 28)):
    """Preprocess image for model inference"""
    try:
        # Load image
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        
        # Resize to target size
        image = image.resize(target_size)
        
        # Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return tensor, image
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None, None

def predict_with_model(model, image_tensor):
    """Make prediction with the model"""
    try:
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            return predicted_class, confidence, probabilities[0].numpy()
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None, None, None

def visualize_results(original_image, predictions, model_name, image_path):
    """Create visualization of results"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Show original image
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title(f'Original Image: {os.path.basename(image_path)}')
    axes[0].axis('off')
    
    # Show prediction results
    classes = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
               'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
               'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
    
    predicted_class, confidence, probabilities = predictions
    
    # Bar chart of top 5 predictions
    top5_indices = np.argsort(probabilities)[-5:][::-1]
    top5_classes = [classes[i] for i in top5_indices]
    top5_probs = [probabilities[i] for i in top5_indices]
    
    axes[1].barh(top5_classes, top5_probs)
    axes[1].set_title(f'{model_name} Predictions\nPredicted: {classes[predicted_class]} ({confidence:.2%})')
    axes[1].set_xlabel('Probability')
    
    plt.tight_layout()
    
    # Save visualization
    output_path = f'test_results_{model_name}_{os.path.basename(image_path)}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return output_path

def main():
    """Main test function"""
    print("=" * 60)
    print("MEDICAL IMAGING AI - COMPLETE PIPELINE TEST")
    print("=" * 60)
    
    # Test images
    test_images = [
        "/Users/user/Downloads/test_medmnist_chest.png",
        "/Users/user/Downloads/test_medical_image.png"
    ]
    
    # Available models (using actual trained models)
    models = {
        'ChestMNIST Model': 'training_results/chestmnist/models/chestmnist_model.pth',
        'Research Paper Model': 'best_research_chestmnist.pth',
        'Advanced CNN (DermaMNIST)': 'best_advanced_dermamnist.pth',
        'EfficientNet (DermaMNIST)': 'best_efficientnet_dermamnist.pth'
    }
    
    results = []
    
    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
            
        print(f"\n📸 Processing: {os.path.basename(image_path)}")
        print("-" * 40)
        
        # Preprocess image
        image_tensor, original_image = preprocess_image(image_path)
        if image_tensor is None:
            continue
            
        print(f"Image shape: {image_tensor.shape}")
        
        # Test with each model
        for model_name, model_path in models.items():
            if not os.path.exists(model_path):
                print(f"Model not found: {model_path}")
                continue
                
            print(f"\n🤖 Testing with {model_name}...")
            
            # Load model
            model = load_trained_model(model_path, model_name.lower().replace(' ', '_'))
            if model is None:
                continue
                
            # Make prediction
            predicted_class, confidence, probabilities = predict_with_model(model, image_tensor)
            if predicted_class is None:
                continue
                
            # Display results
            classes = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
                       'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
                       'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
            
            print(f"  Predicted Class: {classes[predicted_class]}")
            print(f"  Confidence: {confidence:.2%}")
            print(f"  Top 3 Predictions:")
            
            top3_indices = np.argsort(probabilities)[-3:][::-1]
            for i, idx in enumerate(top3_indices):
                print(f"    {i+1}. {classes[idx]}: {probabilities[idx]:.2%}")
            
            # Create visualization
            try:
                viz_path = visualize_results(original_image, 
                                          (predicted_class, confidence, probabilities), 
                                          model_name, image_path)
                print(f"  Visualization saved: {viz_path}")
            except Exception as e:
                print(f"  Visualization error: {e}")
            
            results.append({
                'image': os.path.basename(image_path),
                'model': model_name,
                'predicted_class': classes[predicted_class],
                'confidence': confidence,
                'top3_predictions': [(classes[idx], probabilities[idx]) for idx in top3_indices]
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for result in results:
        print(f"\n📸 {result['image']} - {result['model']}")
        print(f"   Predicted: {result['predicted_class']} ({result['confidence']:.2%})")
        print(f"   Top 3: {', '.join([f'{cls}({prob:.1%})' for cls, prob in result['top3_predictions']])}")
    
    print(f"\n✅ Pipeline test completed! Processed {len(results)} model-image combinations.")
    print("📊 Check the generated visualization files for detailed results.")

if __name__ == "__main__":
    main()
