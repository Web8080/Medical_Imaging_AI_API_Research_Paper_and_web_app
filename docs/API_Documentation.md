# Medical Imaging AI API Documentation

## Overview

The Medical Imaging AI API is a scalable, cloud-based framework that provides plug-and-play tumor detection and measurement capabilities for healthcare applications. The API supports DICOM, NIfTI, and standard image formats, returning precise bounding boxes, segmentation masks, and quantitative metrics.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

Currently, the API operates without authentication for development purposes. In production, implement JWT-based authentication or API key authentication.

## Endpoints

### Health Check

#### GET /health

Check the health status of the API and its dependencies.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2023-01-01T12:00:00Z",
  "version": "1.0.0",
  "uptime": 3600.5
}
```

#### GET /health/detailed

Get detailed health information for all components.

**Response:**
```json
{
  "timestamp": "2023-01-01T12:00:00Z",
  "version": "1.0.0",
  "uptime": 3600.5,
  "components": {
    "database": {
      "status": "healthy",
      "response_time_ms": 5
    },
    "model_service": {
      "status": "healthy",
      "loaded_models": 2,
      "models": ["brain_segmentation", "lung_detection"]
    },
    "redis": {
      "status": "healthy"
    }
  },
  "overall_status": "healthy"
}
```

### File Upload and Processing

#### POST /upload

Upload a medical imaging file for processing.

**Parameters:**
- `file` (required): Medical imaging file (DICOM, NIfTI, JPEG, PNG)
- `model_id` (optional): Specific model to use for processing
- `user_id` (optional): User identifier for tracking

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/upload" \
  -F "file=@brain_scan.dcm" \
  -F "model_id=brain_segmentation" \
  -F "user_id=user123"
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "File uploaded successfully",
  "estimated_processing_time": 30
}
```

**Supported File Formats:**
- DICOM (.dcm, .dicom)
- NIfTI (.nii, .nii.gz)
- JPEG (.jpg, .jpeg)
- PNG (.png)

**File Size Limit:** 500MB

#### GET /upload/jobs/{job_id}

Get the status of a processing job.

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress": 1.0,
  "created_at": "2023-01-01T12:00:00Z",
  "updated_at": "2023-01-01T12:00:30Z",
  "completed_at": "2023-01-01T12:00:30Z",
  "error_message": null
}
```

**Status Values:**
- `pending`: Job is queued for processing
- `processing`: Job is currently being processed
- `completed`: Job completed successfully
- `failed`: Job failed with an error

#### GET /upload/jobs

List processing jobs with optional filtering.

**Query Parameters:**
- `user_id` (optional): Filter by user ID
- `status` (optional): Filter by status
- `limit` (optional): Maximum number of jobs (default: 20)
- `offset` (optional): Number of jobs to skip (default: 0)

**Response:**
```json
[
  {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "completed",
    "created_at": "2023-01-01T12:00:00Z",
    "updated_at": "2023-01-01T12:00:30Z",
    "completed_at": "2023-01-01T12:00:30Z",
    "error_message": null
  }
]
```

## Processing Results

When a job is completed, the results include:

### Detection Result Structure

```json
{
  "detection_id": "seg_1",
  "class_name": "tumor",
  "class_confidence": 0.95,
  "bounding_box": {
    "x_min": 100.0,
    "y_min": 150.0,
    "z_min": 50.0,
    "x_max": 200.0,
    "y_max": 250.0,
    "z_max": 100.0,
    "confidence": 0.95
  },
  "segmentation_mask": {
    "mask_data": "base64_encoded_mask_data",
    "format": "numpy",
    "dimensions": [256, 256, 128],
    "confidence": 0.95
  },
  "metrics": {
    "volume_mm3": 1250.5,
    "surface_area_mm2": 450.2,
    "diameter_mm": 12.5,
    "sphericity": 0.85,
    "elongation": 1.2,
    "compactness": 0.75
  }
}
```

### Complete Processing Result

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "model_used": "brain_segmentation",
  "processing_time_seconds": 2.5,
  "created_at": "2023-01-01T12:00:00Z",
  "completed_at": "2023-01-01T12:00:30Z",
  "detections": [
    {
      "detection_id": "seg_1",
      "class_name": "tumor",
      "class_confidence": 0.95,
      "bounding_box": { ... },
      "segmentation_mask": { ... },
      "metrics": { ... }
    }
  ],
  "summary": {
    "total_detections": 1,
    "average_confidence": 0.95
  },
  "overall_confidence": 0.95,
  "image_quality_score": 0.88,
  "input_metadata": {
    "format": "DICOM",
    "dimensions": [256, 256, 128],
    "voxel_sizes": [1.0, 1.0, 1.0]
  },
  "processing_metadata": {
    "preprocessing_time": 0.5,
    "inference_time": 1.8,
    "postprocessing_time": 0.2
  }
}
```

## Error Handling

The API returns standard HTTP status codes and error messages:

### Error Response Format

```json
{
  "error": "Validation Error",
  "message": "Invalid file format",
  "details": {
    "field": "file",
    "issue": "Unsupported format"
  },
  "timestamp": "2023-01-01T12:00:00Z"
}
```

### Common Error Codes

- `400 Bad Request`: Invalid request parameters or file format
- `404 Not Found`: Job or resource not found
- `413 Payload Too Large`: File size exceeds limit
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error

## Rate Limiting

Currently, no rate limiting is implemented. In production, implement rate limiting based on:
- Requests per minute per user
- File upload size limits
- Concurrent processing limits

## Data Privacy and Compliance

### HIPAA Compliance

- All patient identifying information is automatically removed from DICOM headers
- Data is encrypted in transit and at rest
- Comprehensive audit logging is maintained
- Data retention policies are enforced

### GDPR Compliance

- Users can request data deletion
- Data processing is logged and auditable
- Consent mechanisms are in place
- Data minimization principles are followed

## SDK Examples

### Python

```python
import requests

# Upload file
with open('brain_scan.dcm', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/upload',
        files={'file': f},
        data={'model_id': 'brain_segmentation'}
    )

job_id = response.json()['job_id']

# Check status
status_response = requests.get(f'http://localhost:8000/api/v1/upload/jobs/{job_id}')
print(status_response.json())
```

### JavaScript

```javascript
// Upload file
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('model_id', 'brain_segmentation');

const response = await fetch('http://localhost:8000/api/v1/upload', {
    method: 'POST',
    body: formData
});

const result = await response.json();
const jobId = result.job_id;

// Check status
const statusResponse = await fetch(`http://localhost:8000/api/v1/upload/jobs/${jobId}`);
const status = await statusResponse.json();
console.log(status);
```

### cURL

```bash
# Upload file
curl -X POST "http://localhost:8000/api/v1/upload" \
  -F "file=@brain_scan.dcm" \
  -F "model_id=brain_segmentation"

# Check status
curl "http://localhost:8000/api/v1/upload/jobs/{job_id}"
```

## WebSocket Support (Future)

Real-time processing updates via WebSocket connections:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/jobs/{job_id}');

ws.onmessage = function(event) {
    const update = JSON.parse(event.data);
    console.log('Processing update:', update);
};
```

## Monitoring and Metrics

### Health Monitoring

- API health endpoint for load balancer checks
- Component-specific health checks
- Performance metrics collection

### Metrics Available

- Request count and response times
- Model inference performance
- Error rates and types
- Resource utilization

## Support

For technical support or questions:
- Email: support@medicalimagingai.com
- Documentation: https://docs.medicalimagingai.com
- GitHub Issues: https://github.com/medicalimagingai/api/issues
