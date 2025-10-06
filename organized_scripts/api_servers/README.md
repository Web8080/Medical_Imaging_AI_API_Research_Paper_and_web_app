# 🌐 API Server Scripts

This folder contains API server implementations for the Medical Imaging AI system.

## 📋 Available Scripts

### `simple_api_server.py`
**Purpose**: Simplified API server without complex dependencies
**Use Case**: Quick API testing and demonstration
**Features**:
- FastAPI-based REST API
- Mock model predictions
- Image upload and processing
- CORS support
- Health check endpoints

**Usage**:
```bash
python simple_api_server.py
```

## 🔗 API Endpoints

### Health & Status
- `GET /` - Root endpoint with API information
- `GET /health` - Health check
- `GET /models` - List available models

### Image Processing
- `POST /upload` - Upload and analyze medical image
- `POST /predict` - Alternative prediction endpoint

## 📊 Request/Response Format

### Upload Request
```bash
curl -X POST "http://localhost:8001/upload" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@image.png"
```

### Response Format
```json
{
  "filename": "image.png",
  "file_size": 12345,
  "content_type": "image/png",
  "prediction": {
    "predicted_class": "Infiltration",
    "confidence": 0.85,
    "top5_predictions": [...],
    "model_info": {...}
  },
  "processing_time": "0.1s",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## 🎯 When to Use

- **Development**: Quick API testing
- **Demonstration**: Show API functionality
- **Integration**: Test frontend-backend communication

## 🔧 Configuration

The server runs on:
- **Host**: 0.0.0.0 (all interfaces)
- **Port**: 8001
- **Log Level**: info
