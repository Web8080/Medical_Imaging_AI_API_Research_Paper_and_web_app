# Medical Imaging AI API

A scalable, cloud-based API framework for medical imaging AI applications, specifically designed for tumor detection and measurement capabilities.

## Features

- **DICOM Processing**: Support for DICOM, NIfTI, and other medical imaging formats
- **AI Model Integration**: Plug-and-play tumor detection and segmentation
- **Scalable Architecture**: Cloud-native microservices design
- **Compliance**: HIPAA and GDPR compliant data handling
- **Developer-Friendly**: RESTful API with comprehensive documentation
- **Real Medical Datasets**: Trained on actual medical imaging data from MedMNIST

## Datasets Used

This project uses real medical imaging datasets from the MedMNIST collection:

- **ChestMNIST**: 78,468 chest X-ray images from NIH-ChestXray14 dataset for multi-label disease classification
- **DermaMNIST**: 7,007 dermatoscopic images from HAM10000 dataset for skin lesion classification  
- **OCTMNIST**: 97,477 optical coherence tomography images for retinal disease diagnosis
- **Additional datasets**: BRATS 2021, LIDC-IDRI, Medical Segmentation Decathlon (download scripts provided)

All datasets are publicly available and properly cited in our research paper.

## Quick Start

1. Clone the repository
2. Set up environment variables
3. Install dependencies: `pip install -r requirements.txt`
4. Run the API: `python -m src.main`

## API Endpoints

- `POST /api/v1/upload` - Upload medical images for processing
- `GET /api/v1/jobs/{job_id}` - Retrieve processing results
- `GET /api/v1/models` - List available AI models
- `POST /api/v1/feedback` - Submit feedback on results
- `GET /api/v1/health` - Health check

## Documentation

Full API documentation is available at `/docs` when running the server.

## License

MIT License - see LICENSE file for details.
