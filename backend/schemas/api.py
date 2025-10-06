"""
Pydantic schemas for API requests and responses.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    uptime: float = 0.0


class UploadResponse(BaseModel):
    """Response for file upload."""
    job_id: UUID
    status: str = "pending"
    message: str = "File uploaded successfully"
    estimated_processing_time: Optional[int] = None  # seconds


class ProcessingStatus(BaseModel):
    """Processing job status."""
    job_id: UUID
    status: str  # pending, processing, completed, failed
    progress: Optional[float] = None  # 0.0 to 1.0
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class BoundingBox(BaseModel):
    """Bounding box coordinates."""
    x_min: float = Field(..., description="Minimum x coordinate")
    y_min: float = Field(..., description="Minimum y coordinate")
    z_min: Optional[float] = Field(None, description="Minimum z coordinate (for 3D)")
    x_max: float = Field(..., description="Maximum x coordinate")
    y_max: float = Field(..., description="Maximum y coordinate")
    z_max: Optional[float] = Field(None, description="Maximum z coordinate (for 3D)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")


class SegmentationMask(BaseModel):
    """Segmentation mask information."""
    mask_data: str = Field(..., description="Base64 encoded mask data")
    format: str = Field(default="numpy", description="Mask format")
    dimensions: List[int] = Field(..., description="Mask dimensions")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Segmentation confidence")


class TumorMetrics(BaseModel):
    """Tumor measurement metrics."""
    volume_mm3: Optional[float] = Field(None, description="Volume in cubic millimeters")
    surface_area_mm2: Optional[float] = Field(None, description="Surface area in square millimeters")
    diameter_mm: Optional[float] = Field(None, description="Maximum diameter in millimeters")
    sphericity: Optional[float] = Field(None, ge=0.0, le=1.0, description="Shape sphericity")
    elongation: Optional[float] = Field(None, ge=0.0, description="Shape elongation")
    compactness: Optional[float] = Field(None, ge=0.0, le=1.0, description="Shape compactness")


class DetectionResult(BaseModel):
    """Individual detection result."""
    detection_id: str = Field(..., description="Unique detection identifier")
    class_name: str = Field(..., description="Detected class (e.g., 'tumor', 'lesion')")
    class_confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")
    bounding_box: BoundingBox
    segmentation_mask: Optional[SegmentationMask] = None
    metrics: Optional[TumorMetrics] = None


class ProcessingResult(BaseModel):
    """Complete processing result."""
    job_id: UUID
    status: str = "completed"
    model_used: str
    processing_time_seconds: float
    created_at: datetime
    completed_at: datetime
    
    # Results
    detections: List[DetectionResult] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    
    # Quality metrics
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    image_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Metadata
    input_metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelInfo(BaseModel):
    """Model information."""
    model_id: str
    name: str
    version: str
    type: str  # segmentation, detection, classification
    modality: str  # MRI, CT, X-ray, etc.
    anatomy: str  # brain, lung, liver, etc.
    is_active: bool = True
    
    # Performance metrics
    accuracy: Optional[float] = None
    dice_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    
    # Configuration
    input_size: Optional[List[int]] = None
    supported_formats: List[str] = Field(default_factory=list)
    preprocessing_required: bool = True
    postprocessing_available: bool = True


class FeedbackRequest(BaseModel):
    """User feedback request."""
    job_id: UUID
    rating: Optional[int] = Field(None, ge=1, le=5, description="Rating from 1 to 5")
    feedback_text: Optional[str] = Field(None, max_length=1000)
    is_accurate: Optional[bool] = Field(None, description="Whether the result was accurate")
    feedback_type: str = Field(default="general", description="Type of feedback")


class FeedbackResponse(BaseModel):
    """Feedback submission response."""
    feedback_id: int
    message: str = "Feedback submitted successfully"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = Field(default=1, ge=1, description="Page number")
    size: int = Field(default=20, ge=1, le=100, description="Page size")
    
    @validator('page')
    def validate_page(cls, v):
        if v < 1:
            raise ValueError('Page must be greater than 0')
        return v


class JobListResponse(BaseModel):
    """Response for job listing."""
    jobs: List[ProcessingStatus]
    total: int
    page: int
    size: int
    has_next: bool
    has_prev: bool


class ModelListResponse(BaseModel):
    """Response for model listing."""
    models: List[ModelInfo]
    total: int
    page: int
    size: int
    has_next: bool
    has_prev: bool
