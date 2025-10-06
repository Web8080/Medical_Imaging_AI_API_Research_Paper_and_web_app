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
├── backend/                          # Backend code (main package)
│   ├── __init__.py
│   ├── api/                          # API implementation
│   │   ├── __init__.py
│   │   ├── main.py                   # Main FastAPI application
│   │   ├── simple_api_server.py      # Simplified API server
│   │   ├── v1/                       # API version 1
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
│   │   ├── dataset_downloader.py     # Dataset download scripts
│   │   └── preprocessing.py          # Data preprocessing utilities
│   │
│   ├── visualization/                # Visualization utilities
│   │   ├── __init__.py
│   │   ├── generate_training_visualizations.py # Training visualization
│   │   ├── visualize_advanced_results.py      # Results analysis plots
│   │   └── generate_methodology_comparison.py # Methodology comparison plots
│   │
│   ├── core/                         # Core backend services
│   │   ├── __init__.py
│   │   ├── config.py                 # Configuration management
│   │   ├── database.py               # Database connection
│   │   └── security.py               # Security utilities
│   │
│   ├── services/                     # Business logic services
│   │   ├── __init__.py
│   │   ├── model_service.py          # Model management
│   │   ├── dicom_processor.py        # DICOM processing
│   │   └── job_service.py            # Job management
│   │
│   └── schemas/                      # Data schemas
│       ├── __init__.py
│       └── api.py                    # API schemas
│
├── frontend/                         # Frontend applications
│   ├── streamlit/                    # Streamlit dashboard
│   │   ├── streamlit_dashboard.py    # Main dashboard
│   │   ├── minimal_dashboard.py      # Minimal dashboard variant
│   │   ├── requirements.txt          # Frontend dependencies
│   │   └── README.md                 # Frontend documentation
│   │
│   └── react/                        # React web application
│       ├── package.json
│       ├── src/
│       └── public/
│
├── tests/                            # Test suite
│   ├── test_complete_pipeline.py     # End-to-end tests
│   ├── test_services.py              # Service tests
│   ├── test_api.py                   # API tests
│   ├── test_streamlit.py             # Streamlit tests
│   ├── simple_pipeline_demo.py       # Demo scripts
│   └── conftest.py                   # Test configuration
│
├── scripts/                          # Utility scripts
│   ├── launch_ui.py                  # UI launcher
│   ├── start_web_interface.py        # Web interface starter
│   ├── quick_start.py                # Quick start script
│   └── evaluate_models.py            # Model evaluation script
│
├── docs/                             # Documentation
│   ├── api/                          # API documentation
│   ├── deployment/                   # Deployment guides
│   ├── development/                  # Development guides
│   │   └── PROJECT_STRUCTURE.md      # This file
│   └── audit_reports/                # Project audit reports
│       ├── PROJECT_AUDIT_REPORT.md   # First audit report
│       └── SECOND_AUDIT_REPORT.md    # Second audit report
│
├── assets/                           # Static assets
│   ├── images/                       # Project images
│   ├── icons/                        # Icons and logos
│   ├── test_images/                  # Test images
│   │   ├── test_medmnist_chest.png   # MedMNIST test image
│   │   └── test_medical_image.png    # Medical test image
│   └── UI_UX_Screenshots/            # UI/UX screenshots
│       ├── UI_UX_dashboard.png       # Dashboard screenshot
│       ├── result_history.png        # Results history screenshot
│       └── documentation.png         # Documentation screenshot
│
├── research_paper/                   # Research paper files
│   ├── Medical_Imaging_AI_API_Research_Paper.md  # Research paper (Markdown)
│   └── Medical_Imaging_AI_API_Research_Paper.html # Research paper (HTML)
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
