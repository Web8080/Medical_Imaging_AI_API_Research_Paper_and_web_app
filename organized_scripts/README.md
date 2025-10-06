# 📁 Organized Scripts Directory

This directory contains all Python scripts organized by their use case for easy navigation and maintenance.

## 📂 Folder Structure

### 🔧 `demo_and_testing/`
**Purpose**: Scripts for demonstrating the system and testing functionality
- `simple_pipeline_demo.py` - Complete pipeline demonstration with mock predictions
- `test_complete_pipeline.py` - End-to-end testing with real models
- `test_api.py` - API endpoint testing

### 🚀 `ui_launchers/`
**Purpose**: Scripts to launch the user interface components
- `launch_ui.py` - Main launcher for both Streamlit and API
- `start_frontend.py` - Frontend application starter
- `start_web_interface.py` - Web interface launcher with browser opening

### 🌐 `api_servers/`
**Purpose**: API server implementations
- `simple_api_server.py` - Simplified API server without complex dependencies

### 📊 `data_processing/`
**Purpose**: Scripts for downloading and processing medical datasets
- `download_direct_datasets.py` - Direct dataset downloads
- `download_real_datasets.py` - Real medical dataset downloads
- `download_real_medical_data.py` - Medical data acquisition
- `create_synthetic_data.py` - Synthetic data generation

### 🤖 `model_training/`
**Purpose**: Scripts for training AI models
- `train_medmnist_models.py` - MedMNIST dataset training
- `train_advanced_models.py` - Advanced CNN and EfficientNet training
- `train_research_paper_methodology.py` - Research paper methodology implementation
- `train_models.py` - General model training

### 📈 `visualization/`
**Purpose**: Scripts for generating plots, charts, and visualizations
- `generate_training_visualizations.py` - Training result visualizations
- `generate_advanced_training_plots.py` - Advanced model plots
- `generate_methodology_comparison.py` - Methodology comparison charts
- `generate_research_paper_plots.py` - Research paper specific plots
- `generate_simple_medmnist_plots.py` - MedMNIST training plots
- `generate_real_medmnist_summary.py` - MedMNIST summary reports
- `visualize_advanced_results.py` - Advanced results visualization
- `compare_methodologies.py` - Methodology comparison analysis

## 🎯 Quick Start Guide

### For Demo/Testing:
```bash
cd demo_and_testing/
python simple_pipeline_demo.py
```

### For UI Launch:
```bash
cd ui_launchers/
python launch_ui.py
```

### For Model Training:
```bash
cd model_training/
python train_medmnist_models.py
```

### For Visualizations:
```bash
cd visualization/
python generate_training_visualizations.py
```

## 📝 Notes

- All scripts maintain their original functionality
- Scripts are organized by primary use case
- Each folder has a clear, memorable purpose
- Original file locations are preserved in the main project structure
