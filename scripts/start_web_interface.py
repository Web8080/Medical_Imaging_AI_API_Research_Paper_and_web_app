#!/usr/bin/env python3
"""
Medical Imaging AI - Web Interface Starter
Simple script to start the web interface and open browser
"""

import webbrowser
import time
import os

def main():
    """Start the web interface"""
    print("=" * 60)
    print("🏥 MEDICAL IMAGING AI - WEB INTERFACE")
    print("=" * 60)
    
    # Streamlit Dashboard URL
    streamlit_url = "http://localhost:8501"
    
    print("🚀 Starting Medical Imaging AI Web Interface...")
    print("")
    print("📊 STREAMLIT DASHBOARD:")
    print(f"   URL: {streamlit_url}")
    print("   Features:")
    print("   • Upload medical images (DICOM, PNG, JPG)")
    print("   • AI-powered analysis and predictions")
    print("   • Interactive results visualization")
    print("   • DICOM image viewer with tools")
    print("   • Results history and comparison")
    print("")
    
    # Open browser
    try:
        print("🌐 Opening browser...")
        webbrowser.open(streamlit_url)
        print("✅ Browser opened successfully!")
    except Exception as e:
        print(f"❌ Error opening browser: {e}")
        print(f"Please manually open: {streamlit_url}")
    
    print("")
    print("=" * 60)
    print("🎯 HOW TO USE THE INTERFACE")
    print("=" * 60)
    print("1. 📸 Upload a medical image using the file uploader")
    print("2. 🤖 Select an AI model for analysis")
    print("3. ⚡ Click 'Analyze Image' to process")
    print("4. 📊 View results, predictions, and visualizations")
    print("5. 🔍 Use DICOM viewer tools for detailed inspection")
    print("6. 📋 Check results history for previous analyses")
    print("")
    
    print("=" * 60)
    print("🔧 AVAILABLE AI MODELS")
    print("=" * 60)
    print("• Advanced CNN - High accuracy for medical imaging")
    print("• EfficientNet - Optimized for mobile/edge deployment")
    print("• Research Paper Model - U-Net inspired architecture")
    print("")
    
    print("=" * 60)
    print("📊 SUPPORTED IMAGE FORMATS")
    print("=" * 60)
    print("• DICOM (.dcm) - Medical imaging standard")
    print("• PNG (.png) - High quality images")
    print("• JPEG (.jpg, .jpeg) - Compressed images")
    print("• NIfTI (.nii) - Neuroimaging format")
    print("")
    
    print("✅ Web interface is ready to use!")
    print("🎉 Enjoy analyzing medical images with AI!")

if __name__ == "__main__":
    main()
