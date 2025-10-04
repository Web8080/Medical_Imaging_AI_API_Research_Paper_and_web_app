"""
API endpoint tests.
"""

import pytest
from fastapi.testclient import TestClient


def test_root_endpoint(client: TestClient):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert data["message"] == "Medical Imaging AI API"


def test_health_check(client: TestClient):
    """Test health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "timestamp" in data
    assert "version" in data
    assert "uptime" in data


def test_detailed_health_check(client: TestClient):
    """Test detailed health check endpoint."""
    response = client.get("/api/v1/health/detailed")
    assert response.status_code == 200
    data = response.json()
    assert "timestamp" in data
    assert "version" in data
    assert "uptime" in data
    assert "components" in data
    assert "overall_status" in data


def test_upload_endpoint_no_file(client: TestClient):
    """Test upload endpoint without file."""
    response = client.post("/api/v1/upload")
    assert response.status_code == 422  # Validation error


def test_upload_endpoint_with_file(client: TestClient, sample_image_file):
    """Test upload endpoint with file."""
    with open(sample_image_file, "rb") as f:
        response = client.post(
            "/api/v1/upload",
            files={"file": ("test.png", f, "image/png")}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert "status" in data
    assert "message" in data
    assert data["status"] == "pending"


def test_upload_endpoint_with_dicom(client: TestClient, sample_dicom_file):
    """Test upload endpoint with DICOM file."""
    with open(sample_dicom_file, "rb") as f:
        response = client.post(
            "/api/v1/upload",
            files={"file": ("test.dcm", f, "application/dicom")}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert "status" in data
    assert data["status"] == "pending"


def test_get_job_status_not_found(client: TestClient):
    """Test getting status of non-existent job."""
    response = client.get("/api/v1/upload/jobs/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


def test_list_jobs(client: TestClient):
    """Test listing jobs."""
    response = client.get("/api/v1/upload/jobs")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_list_jobs_with_filters(client: TestClient):
    """Test listing jobs with filters."""
    response = client.get("/api/v1/upload/jobs?status=pending&limit=10&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_upload_with_model_id(client: TestClient, sample_image_file):
    """Test upload with model ID specified."""
    with open(sample_image_file, "rb") as f:
        response = client.post(
            "/api/v1/upload",
            files={"file": ("test.png", f, "image/png")},
            data={"model_id": "test_model"}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "pending"


def test_upload_with_user_id(client: TestClient, sample_image_file):
    """Test upload with user ID specified."""
    with open(sample_image_file, "rb") as f:
        response = client.post(
            "/api/v1/upload",
            files={"file": ("test.png", f, "image/png")},
            data={"user_id": "test_user"}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "pending"


def test_upload_invalid_file_type(client: TestClient):
    """Test upload with invalid file type."""
    # Create a text file
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_file:
        tmp_file.write(b"This is not a medical image")
        tmp_file.flush()
        
        with open(tmp_file.name, "rb") as f:
            response = client.post(
                "/api/v1/upload",
                files={"file": ("test.txt", f, "text/plain")}
            )
    
    assert response.status_code == 400


def test_upload_empty_file(client: TestClient):
    """Test upload with empty file."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        # Empty file
        pass
        
        with open(tmp_file.name, "rb") as f:
            response = client.post(
                "/api/v1/upload",
                files={"file": ("empty.png", f, "image/png")}
            )
    
    assert response.status_code == 400
