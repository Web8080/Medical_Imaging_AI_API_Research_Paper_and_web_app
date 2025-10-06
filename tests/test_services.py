"""
Service layer tests.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from backend.services.dicom_processor import DICOMProcessor
from backend.services.model_service import ModelService
from backend.services.job_service import JobService


class TestDICOMProcessor:
    """Test DICOM processor service."""
    
    def test_init(self):
        """Test processor initialization."""
        processor = DICOMProcessor()
        assert processor.supported_formats is not None
        assert processor.max_file_size > 0
    
    def test_detect_format_by_extension(self):
        """Test format detection by file extension."""
        processor = DICOMProcessor()
        
        # Test DICOM
        assert processor._detect_format("test.dcm", b"") == "DICOM"
        assert processor._detect_format("test.dicom", b"") == "DICOM"
        
        # Test NIfTI
        assert processor._detect_format("test.nii", b"") == "NIfTI"
        assert processor._detect_format("test.nii.gz", b"") == "NIfTI"
        
        # Test images
        assert processor._detect_format("test.jpg", b"") == "JPEG"
        assert processor._detect_format("test.png", b"") == "PNG"
    
    def test_detect_format_by_content(self):
        """Test format detection by content."""
        processor = DICOMProcessor()
        
        # Test DICOM magic bytes
        dicom_content = b"DICM" + b"0" * 100
        assert processor._detect_format("unknown", dicom_content) == "DICOM"
        
        # Test NIfTI magic bytes
        nifti_content = b"\x93\x4E\x49\x49\x1A\x0A\x00\x00" + b"0" * 100
        assert processor._detect_format("unknown", nifti_content) == "NIfTI"
        
        # Test JPEG magic bytes
        jpeg_content = b"\xFF\xD8\xFF" + b"0" * 100
        assert processor._detect_format("unknown", jpeg_content) == "JPEG"
        
        # Test PNG magic bytes
        png_content = b"\x89PNG\r\n\x1a\n" + b"0" * 100
        assert processor._detect_format("unknown", png_content) == "PNG"
    
    def test_calculate_quality_score(self):
        """Test image quality score calculation."""
        processor = DICOMProcessor()
        
        # Test with random image
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        score = processor._calculate_quality_score(image)
        
        assert 0.0 <= score <= 1.0
        assert isinstance(score, float)
    
    def test_normalize_intensity(self):
        """Test intensity normalization."""
        processor = DICOMProcessor()
        
        # Test with random data
        data = np.random.randint(0, 1000, (50, 50))
        normalized = processor._normalize_intensity(data)
        
        assert normalized.shape == data.shape
        assert np.isclose(np.mean(normalized), 0.0, atol=1e-10)
        assert np.isclose(np.std(normalized), 1.0, atol=1e-10)
    
    def test_resize_to_target(self):
        """Test image resizing."""
        processor = DICOMProcessor()
        
        # Test 2D image
        image_2d = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        resized = processor._resize_to_target(image_2d, [50, 50])
        
        assert resized.shape == (50, 50)
        
        # Test 3D image
        image_3d = np.random.randint(0, 255, (10, 100, 100), dtype=np.uint8)
        resized_3d = processor._resize_to_target(image_3d, [5, 50, 50])
        
        assert resized_3d.shape == (5, 50, 50)


class TestModelService:
    """Test model service."""
    
    def test_init(self):
        """Test model service initialization."""
        service = ModelService()
        assert service.models == {}
        assert service.cache_size > 0
    
    def test_load_model_pytorch(self):
        """Test loading PyTorch model."""
        service = ModelService()
        
        config = {
            "type": "pytorch",
            "task_type": "segmentation",
            "input_size": [256, 256]
        }
        
        # Mock model path
        with patch("torch.load") as mock_load:
            mock_load.return_value = {}
            result = service.load_model("test_model", "dummy_path", config)
        
        assert result is True
        assert "test_model" in service.models
    
    def test_get_available_models(self):
        """Test getting available models."""
        service = ModelService()
        
        # Add a mock model
        service.models["test_model"] = {
            "model": Mock(),
            "config": {"type": "pytorch"},
            "loaded_at": 1234567890
        }
        
        models = service.get_available_models()
        assert len(models) == 1
        assert models[0]["model_id"] == "test_model"
    
    def test_unload_model(self):
        """Test unloading model."""
        service = ModelService()
        
        # Add a mock model
        service.models["test_model"] = {
            "model": Mock(),
            "config": {"type": "pytorch"},
            "loaded_at": 1234567890
        }
        
        result = service.unload_model("test_model")
        assert result is True
        assert "test_model" not in service.models
        
        # Test unloading non-existent model
        result = service.unload_model("non_existent")
        assert result is False
    
    def test_prepare_input(self):
        """Test input preparation."""
        service = ModelService()
        
        # Test 2D image
        image_2d = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        tensor = service._prepare_input(image_2d, {})
        
        assert tensor.shape == (1, 1, 100, 100)  # (batch, channel, height, width)
        
        # Test 3D image
        image_3d = np.random.randint(0, 255, (10, 100, 100), dtype=np.uint8)
        tensor_3d = service._prepare_input(image_3d, {})
        
        assert tensor_3d.shape == (1, 1, 10, 100, 100)
    
    def test_calculate_bounding_box_2d(self):
        """Test 2D bounding box calculation."""
        service = ModelService()
        
        # Create a mask with a square region
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 30:50] = 1
        
        bbox = service._calculate_bounding_box(mask)
        
        assert bbox.x_min == 30
        assert bbox.y_min == 20
        assert bbox.x_max == 49
        assert bbox.y_max == 39
        assert bbox.z_min is None
        assert bbox.z_max is None
    
    def test_calculate_bounding_box_3d(self):
        """Test 3D bounding box calculation."""
        service = ModelService()
        
        # Create a 3D mask with a cube region
        mask = np.zeros((50, 100, 100), dtype=np.uint8)
        mask[10:20, 20:40, 30:50] = 1
        
        bbox = service._calculate_bounding_box(mask)
        
        assert bbox.x_min == 30
        assert bbox.y_min == 20
        assert bbox.z_min == 10
        assert bbox.x_max == 49
        assert bbox.y_max == 39
        assert bbox.z_max == 19
    
    def test_calculate_tumor_metrics(self):
        """Test tumor metrics calculation."""
        service = ModelService()
        
        # Create a simple mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 1  # 20x20 square
        
        original_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        metrics = service._calculate_tumor_metrics(mask, original_image)
        
        assert metrics.volume_mm3 == 400  # 20 * 20
        assert metrics.surface_area_mm2 > 0
        assert metrics.diameter_mm > 0
        assert 0.0 <= metrics.sphericity <= 1.0
        assert metrics.elongation >= 0.0
        assert 0.0 <= metrics.compactness <= 1.0


class TestJobService:
    """Test job service."""
    
    def test_init(self):
        """Test job service initialization."""
        service = JobService()
        assert service is not None
    
    def test_update_job_status(self):
        """Test updating job status in cache."""
        service = JobService()
        mock_redis = Mock()
        
        service.update_job_status(mock_redis, "test_job", "completed")
        
        mock_redis.setex.assert_called_once()
    
    def test_list_jobs(self):
        """Test listing jobs."""
        service = JobService()
        mock_db = Mock()
        
        # Mock query result
        mock_job = Mock()
        mock_job.id = "test_job"
        mock_job.status = "completed"
        mock_job.created_at = "2023-01-01T00:00:00"
        mock_job.updated_at = "2023-01-01T00:00:00"
        mock_job.completed_at = "2023-01-01T00:00:00"
        mock_job.error_message = None
        
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.limit.return_value = [mock_job]
        
        mock_db.query.return_value = mock_query
        
        jobs = service.list_jobs(mock_db)
        
        assert len(jobs) == 1
        assert jobs[0].job_id == "test_job"
        assert jobs[0].status == "completed"
    
    def test_get_job_results(self):
        """Test getting job results."""
        service = JobService()
        mock_db = Mock()
        
        # Mock job
        mock_job = Mock()
        mock_job.id = "test_job"
        mock_job.status = "completed"
        mock_job.results = {"test": "data"}
        mock_job.confidence_scores = {"confidence": 0.95}
        mock_job.model_used = "test_model"
        mock_job.processing_time_seconds = 1.5
        mock_job.created_at = "2023-01-01T00:00:00"
        mock_job.completed_at = "2023-01-01T00:00:00"
        mock_job.error_message = None
        
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = mock_job
        
        mock_db.query.return_value = mock_query
        
        results = service.get_job_results(mock_db, "test_job")
        
        assert results is not None
        assert results["job_id"] == "test_job"
        assert results["status"] == "completed"
        assert results["results"] == {"test": "data"}
    
    def test_delete_job(self):
        """Test deleting job."""
        service = JobService()
        mock_db = Mock()
        
        # Mock job
        mock_job = Mock()
        mock_job.id = "test_job"
        
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = mock_job
        
        mock_db.query.return_value = mock_query
        
        result = service.delete_job(mock_db, "test_job")
        
        assert result is True
        mock_db.delete.assert_called_once_with(mock_job)
        mock_db.commit.assert_called_once()
