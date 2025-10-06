#!/usr/bin/env python3
"""
Simple Pipeline Demo
Demonstrates the Medical Imaging AI pipeline with mock predictions
"""

import os
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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

def mock_prediction(image_tensor):
    """Mock prediction function (simulates model inference)"""
    # Simulate model prediction with random but realistic probabilities
    np.random.seed(42)  # For reproducible results
    
    # ChestMNIST classes
    classes = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
               'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
               'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
    
    # Generate realistic probabilities (some classes more likely than others)
    base_probs = np.array([0.05, 0.08, 0.12, 0.15, 0.06, 0.10, 0.09, 0.04, 
                          0.11, 0.07, 0.03, 0.05, 0.08, 0.02])
    
    # Add some noise
    noise = np.random.normal(0, 0.02, len(classes))
    probabilities = np.clip(base_probs + noise, 0, 1)
    
    # Normalize to sum to 1
    probabilities = probabilities / probabilities.sum()
    
    # Get prediction
    predicted_class = np.argmax(probabilities)
    confidence = probabilities[predicted_class]
    
    return predicted_class, confidence, probabilities

def visualize_results(original_image, predictions, model_name, image_path):
    """Create visualization of results"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Show original image
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title(f'Original Image: {os.path.basename(image_path)}', fontsize=12)
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
    
    bars = axes[1].barh(top5_classes, top5_probs, color='skyblue', alpha=0.7)
    axes[1].set_title(f'{model_name} Predictions\nPredicted: {classes[predicted_class]} ({confidence:.1%})', 
                     fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Probability', fontsize=10)
    axes[1].set_xlim(0, max(top5_probs) * 1.1)
    
    # Add probability values on bars
    for i, (bar, prob) in enumerate(zip(bars, top5_probs)):
        axes[1].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
                    f'{prob:.1%}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save visualization
    base_name = os.path.splitext(os.path.basename(image_path))[0]  # Remove extension
    output_path = f'demo_results_{model_name.replace(" ", "_")}_{base_name}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  📊 Visualization saved: {output_path}")
    
    return output_path

def main():
    """Main demo function"""
    print("=" * 70)
    print("🏥 MEDICAL IMAGING AI - PIPELINE DEMONSTRATION")
    print("=" * 70)
    
    # Test images
    test_images = [
        "test_medmnist_chest.png",
        "test_medical_image.png"
    ]
    
    # Simulated models
    models = {
        'Advanced CNN': 'Simulated Advanced CNN with attention mechanisms',
        'EfficientNet': 'Simulated EfficientNet with mobile optimization',
        'Research Paper Model': 'Simulated U-Net inspired architecture'
    }
    
    results = []
    
    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            continue
            
        print(f"\n📸 Processing: {os.path.basename(image_path)}")
        print("-" * 50)
        
        # Preprocess image
        image_tensor, original_image = preprocess_image(image_path)
        if image_tensor is None:
            continue
            
        print(f"✅ Image preprocessed - Shape: {image_tensor.shape}")
        
        # Test with each model
        for model_name, model_desc in models.items():
            print(f"\n🤖 Testing with {model_name}...")
            print(f"   Description: {model_desc}")
            
            # Make mock prediction
            predicted_class, confidence, probabilities = mock_prediction(image_tensor)
            
            # Display results
            classes = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
                       'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
                       'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
            
            print(f"   🎯 Predicted Class: {classes[predicted_class]}")
            print(f"   📊 Confidence: {confidence:.1%}")
            print(f"   🔝 Top 3 Predictions:")
            
            top3_indices = np.argsort(probabilities)[-3:][::-1]
            for i, idx in enumerate(top3_indices):
                print(f"      {i+1}. {classes[idx]}: {probabilities[idx]:.1%}")
            
            # Create visualization
            try:
                viz_path = visualize_results(original_image, 
                                          (predicted_class, confidence, probabilities), 
                                          model_name, image_path)
            except Exception as e:
                print(f"   ❌ Visualization error: {e}")
                viz_path = None
            
            results.append({
                'image': os.path.basename(image_path),
                'model': model_name,
                'predicted_class': classes[predicted_class],
                'confidence': confidence,
                'top3_predictions': [(classes[idx], probabilities[idx]) for idx in top3_indices],
                'visualization': viz_path
            })
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 DEMONSTRATION SUMMARY")
    print("=" * 70)
    
    for result in results:
        print(f"\n📸 {result['image']} - {result['model']}")
        print(f"   🎯 Predicted: {result['predicted_class']} ({result['confidence']:.1%})")
        print(f"   🔝 Top 3: {', '.join([f'{cls}({prob:.1%})' for cls, prob in result['top3_predictions']])}")
        if result['visualization']:
            print(f"   📊 Chart: {result['visualization']}")
    
    print(f"\n✅ Pipeline demonstration completed!")
    print(f"📊 Processed {len(results)} model-image combinations")
    print(f"🎨 Generated {len([r for r in results if r['visualization']])} visualization charts")
    
    print("\n" + "=" * 70)
    print("🚀 NEXT STEPS:")
    print("=" * 70)
    print("1. 🔧 Integrate with real trained models")
    print("2. 🌐 Deploy API server for production use")
    print("3. 📱 Launch web interface for user interaction")
    print("4. 🔒 Add security and authentication")
    print("5. 📈 Implement monitoring and analytics")

if __name__ == "__main__":
    main()
