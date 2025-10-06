# Project Structure

This document outlines the professional structure of the Medical Imaging AI API project.

## Directory Structure

```
Medical_Imaging_AI_API/
├── README.md                          # Main project documentation
├── LICENSE                            # Project license
├── .gitignore                         # Git ignore rules
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package setup configuration
├── docker-compose.yml                # Docker orchestration
├── Dockerfile                        # Docker container definition
│
├── src/                              # Source code (main package)
│   ├── __init__.py
│   ├── api/                          # API implementation
│   │   ├── __init__.py
│   │   ├── main.py                   # Main FastAPI application
│   │   ├── simple_api_server.py      # Simplified API server
│   │   ├── models/                   # API model definitions
│   │   ├── preprocessing/            # Data preprocessing
│   │   └── utils/                    # API utilities
│   │
│   ├── models/                       # AI model implementations
│   │   ├── __init__.py
│   │   ├── simple_cnn.py             # Simple CNN model
│   │   ├── advanced_cnn.py           # Advanced CNN model
│   │   ├── efficientnet.py           # EfficientNet model
│   │   └── research_paper_cnn.py     # Research paper methodology
│   │
│   ├── data/                         # Data processing
│   │   ├── __init__.py
│   │   ├── download_datasets.py      # Dataset download scripts
│   │   └── preprocessing.py          # Data preprocessing utilities
│   │
│   └── visualization/                # Visualization utilities
│       ├── __init__.py
│       ├── training_plots.py         # Training visualization
│       ├── results_analysis.py       # Results analysis plots
│       └── methodology_comparison.py # Methodology comparison plots
│
├── frontend/                         # Frontend applications
│   ├── streamlit_dashboard.py        # Streamlit dashboard
│   ├── requirements.txt              # Frontend dependencies
│   └── react-app/                    # React web application
│       ├── package.json
│       ├── src/
│       └── public/
│
├── tests/                            # Test suite
│   ├── unit/                         # Unit tests
│   ├── integration/                  # Integration tests
│   ├── test_complete_pipeline.py     # End-to-end tests
│   └── simple_pipeline_demo.py       # Demo scripts
│
├── scripts/                          # Utility scripts
│   ├── launch_ui.py                  # UI launcher
│   ├── start_web_interface.py        # Web interface starter
│   └── quick_start.py                # Quick start script
│
├── docs/                             # Documentation
│   ├── api/                          # API documentation
│   ├── deployment/                   # Deployment guides
│   ├── development/                  # Development guides
│   └── PROJECT_STRUCTURE.md          # This file
│
├── assets/                           # Static assets
│   ├── images/                       # Project images
│   ├── icons/                        # Icons and logos
│   └── UI_UX_Screenshots/            # UI/UX screenshots
│
├── results/                          # Training results
│   ├── real_medmnist_training/       # MedMNIST training results
│   ├── advanced_training/            # Advanced model results
│   └── research_paper_training/      # Research paper methodology results
│
├── training_results/                 # Organized training outputs
│   ├── chestmnist/                   # ChestMNIST results
│   ├── octmnist/                     # OCTMNIST results
│   ├── advanced_models/              # Advanced model results
│   ├── methodology_comparison/       # Methodology comparison
│   └── research_paper_visualizations/ # Research paper visualizations
│
└── research_paper_implementation/    # Research paper methodology
    ├── README.md
    └── scripts/
```

## Key Principles

### 1. **Separation of Concerns**
- `src/` contains all source code organized by functionality
- `tests/` contains all testing code
- `docs/` contains all documentation
- `assets/` contains static resources

### 2. **Professional Naming**
- Clear, descriptive directory names
- No temporary or vague names like "organized_scripts"
- Follows Python package conventions

### 3. **Modularity**
- Each directory has a specific purpose
- Proper `__init__.py` files for Python packages
- Clear boundaries between different components

### 4. **Scalability**
- Structure supports future growth
- Easy to add new modules or features
- Clear separation between development and production code

### 5. **Industry Standards**
- Follows Python packaging best practices
- Compatible with CI/CD pipelines
- Easy to deploy and maintain

## Migration from Previous Structure

The previous `organized_scripts/` folder has been restructured as follows:

- `organized_scripts/api_servers/` → `src/api/`
- `organized_scripts/model_training/` → `src/models/`
- `organized_scripts/data_processing/` → `src/data/`
- `organized_scripts/visualization/` → `src/visualization/`
- `organized_scripts/demo_and_testing/` → `tests/`
- `organized_scripts/ui_launchers/` → `scripts/`

This provides a much more professional and maintainable structure.
