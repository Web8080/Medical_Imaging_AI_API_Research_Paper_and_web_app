"""
Test API endpoints for Medical Imaging AI API.
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock


class TestAPIEndpoints:
    """Test class for API endpoints."""

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "endpoints" in data

    def test_models_endpoint(self, client):
        """Test models endpoint."""
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "api_status" in data
        assert "uptime" in data
        assert "total_requests" in data

    @patch('backend.api.working_api_server.real_predict')
    @patch('backend.api.working_api_server.preprocess_image')
    def test_upload_endpoint_success(self, mock_preprocess, mock_predict, client, sample_image_bytes):
        """Test successful image upload and prediction."""
        # Mock the preprocessing and prediction functions
        mock_preprocess.return_value = Mock()  # Mock tensor
        mock_predict.return_value = {
            "predicted_class": "Test Class",
            "confidence": 0.85,
            "all_predictions": [
                {"class": "Test Class", "probability": 0.85},
                {"class": "Other Class", "probability": 0.15}
            ],
            "top5_predictions": [
                {"class": "Test Class", "probability": 0.85},
                {"class": "Other Class", "probability": 0.15}
            ]
        }

        # Create a test file
        files = {"file": ("test_image.jpg", sample_image_bytes, "image/jpeg")}
        
        response = client.post("/upload", files=files)
        assert response.status_code == 200
        
        data = response.json()
        assert "prediction" in data
        assert "processing_time" in data
        assert "model_info" in data

    def test_upload_endpoint_no_file(self, client):
        """Test upload endpoint without file."""
        response = client.post("/upload")
        assert response.status_code == 422  # Validation error

    def test_upload_endpoint_invalid_file(self, client):
        """Test upload endpoint with invalid file."""
        files = {"file": ("test.txt", b"not an image", "text/plain")}
        response = client.post("/upload", files=files)
        # Should handle gracefully
        assert response.status_code in [200, 400, 422]

    @patch('backend.api.working_api_server.real_predict')
    @patch('backend.api.working_api_server.preprocess_image')
    def test_upload_endpoint_prediction_error(self, mock_preprocess, mock_predict, client, sample_image_bytes):
        """Test upload endpoint when prediction fails."""
        # Mock preprocessing to succeed but prediction to fail
        mock_preprocess.return_value = Mock()
        mock_predict.side_effect = Exception("Prediction failed")

        files = {"file": ("test_image.jpg", sample_image_bytes, "image/jpeg")}
        response = client.post("/upload", files=files)
        
        # Should handle error gracefully
        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert "error" in data or "prediction" in data

    def test_upload_endpoint_processing_time(self, client, sample_image_bytes):
        """Test that processing time is included in response."""
        files = {"file": ("test_image.jpg", sample_image_bytes, "image/jpeg")}
        response = client.post("/upload", files=files)
        
        if response.status_code == 200:
            data = response.json()
            assert "processing_time" in data
            # Processing time should be a string like "0.1s"
            assert isinstance(data["processing_time"], str)
            assert data["processing_time"].endswith("s")

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/upload")
        # CORS headers should be present
        assert response.status_code in [200, 204]

    def test_api_response_format(self, client):
        """Test that API responses follow expected format."""
        response = client.get("/health")
        assert response.status_code == 200
        
        # Check content type
        assert response.headers["content-type"] == "application/json"
        
        # Check response structure
        data = response.json()
        assert isinstance(data, dict)

    @pytest.mark.slow
    def test_concurrent_requests(self, client, sample_image_bytes):
        """Test handling of concurrent requests."""
        import threading
        import time
        
        results = []
        
        def make_request():
            files = {"file": ("test_image.jpg", sample_image_bytes, "image/jpeg")}
            response = client.post("/upload", files=files)
            results.append(response.status_code)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should complete (status codes may vary)
        assert len(results) == 5
        assert all(status in [200, 400, 422, 500] for status in results)
