"""
Pytest configuration and fixtures.
"""

import os
import tempfile
from typing import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.main import app
from src.core.database import get_db, Base
from src.core.config import settings


# Test database URL
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

# Create test engine
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing."""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


@pytest.fixture(scope="session")
def db_engine():
    """Create test database engine."""
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def db_session(db_engine):
    """Create test database session."""
    connection = db_engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(scope="function")
def client(db_session):
    """Create test client."""
    app.dependency_overrides[get_db] = lambda: db_session
    
    with TestClient(app) as test_client:
        yield test_client
    
    app.dependency_overrides.clear()


@pytest.fixture
def sample_dicom_file():
    """Create a sample DICOM file for testing."""
    import pydicom
    import numpy as np
    from pydicom.dataset import Dataset, FileDataset
    
    # Create a simple DICOM dataset
    ds = Dataset()
    ds.PatientName = "Test^Patient"
    ds.PatientID = "TEST001"
    ds.StudyDate = "20230101"
    ds.StudyTime = "120000"
    ds.Modality = "MR"
    ds.SeriesDescription = "Test Series"
    ds.Rows = 256
    ds.Columns = 256
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1
    ds.PlanarConfiguration = 0
    
    # Create pixel data
    pixel_array = np.random.randint(0, 4096, (256, 256), dtype=np.uint16)
    ds.PixelData = pixel_array.tobytes()
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp_file:
        ds.save_as(tmp_file.name)
        yield tmp_file.name
    
    # Clean up
    os.unlink(tmp_file.name)


@pytest.fixture
def sample_image_file():
    """Create a sample image file for testing."""
    from PIL import Image
    import numpy as np
    
    # Create a simple test image
    image_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    image = Image.fromarray(image_array)
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        image.save(tmp_file.name)
        yield tmp_file.name
    
    # Clean up
    os.unlink(tmp_file.name)


@pytest.fixture
def mock_model_service():
    """Mock model service for testing."""
    from unittest.mock import Mock
    
    mock_service = Mock()
    mock_service.get_available_models.return_value = [
        {
            "model_id": "test_model",
            "config": {
                "type": "pytorch",
                "task_type": "segmentation",
                "input_size": [256, 256]
            },
            "loaded_at": 1234567890
        }
    ]
    
    return mock_service


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    from unittest.mock import Mock
    
    mock_redis = Mock()
    mock_redis.get.return_value = None
    mock_redis.setex.return_value = True
    mock_redis.ping.return_value = True
    
    return mock_redis
