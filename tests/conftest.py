"""
Pytest configuration and fixtures for Medical Imaging AI API tests.
"""

import pytest
import asyncio
import sys
import os
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the API server
try:
    from backend.api.working_api_server import app
except ImportError:
    # Fallback for different import paths
    sys.path.append(str(project_root / "backend"))
    from api.working_api_server import app


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    """Create sample image bytes for testing."""
    # This would be replaced with actual test image data
    return b"fake_image_data"


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    mock_model = Mock()
    mock_model.eval.return_value = None
    mock_model.return_value = Mock()
    return mock_model


@pytest.fixture
def sample_prediction_response():
    """Sample prediction response for testing."""
    return {
        "predicted_class": "Test Class",
        "confidence": 0.85,
        "all_predictions": [
            {"class": "Test Class", "probability": 0.85},
            {"class": "Other Class", "probability": 0.15}
        ],
        "top5_predictions": [
            {"class": "Test Class", "probability": 0.85},
            {"class": "Other Class", "probability": 0.15}
        ],
        "model_info": {
            "name": "Test Model",
            "type": "test",
            "version": "1.0.0",
            "description": "Test model for unit testing"
        }
    }


@pytest.fixture
def sample_metrics_response():
    """Sample metrics response for testing."""
    return {
        "api_status": "healthy",
        "models_loaded": True,
        "uptime": "0:05:30",
        "total_requests": 10,
        "successful_requests": 9,
        "failed_requests": 1,
        "error_rate": 0.1,
        "average_response_time": "0.25s",
        "memory_usage": "45.2%",
        "cpu_usage": "12.8%"
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment before each test."""
    # Set test environment variables
    os.environ["TESTING"] = "true"
    yield
    # Cleanup after test
    if "TESTING" in os.environ:
        del os.environ["TESTING"]


@pytest.fixture
def temp_model_file(tmp_path):
    """Create a temporary model file for testing."""
    model_file = tmp_path / "test_model.pth"
    model_file.write_bytes(b"fake_model_data")
    return str(model_file)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests as API tests"
    )
    config.addinivalue_line(
        "markers", "model: marks tests as model tests"
    )