"""
Database models for storing metadata and processing results.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Boolean, Float
from sqlalchemy.dialects.postgresql import UUID
import uuid

from ..core.database import Base


class ProcessingJob(Base):
    """Model for tracking processing jobs."""
    
    __tablename__ = "processing_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=True)  # Optional user identification
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # File information
    original_filename = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    file_format = Column(String(50), nullable=False)
    file_hash = Column(String(64), nullable=False)  # SHA-256 hash for deduplication
    
    # Processing information
    model_used = Column(String(100), nullable=True)
    processing_time_seconds = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Results
    results = Column(JSON, nullable=True)  # Store processing results as JSON
    confidence_scores = Column(JSON, nullable=True)  # Model confidence scores
    
    # Compliance and audit
    anonymized = Column(Boolean, default=True)
    retention_date = Column(DateTime, nullable=True)  # For GDPR compliance


class ModelMetadata(Base):
    """Model for storing AI model metadata and versions."""
    
    __tablename__ = "model_metadata"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), unique=True, nullable=False)
    model_version = Column(String(50), nullable=False)
    model_type = Column(String(50), nullable=False)  # segmentation, detection, classification
    modality = Column(String(50), nullable=False)  # MRI, CT, X-ray, etc.
    anatomy = Column(String(100), nullable=False)  # brain, lung, liver, etc.
    
    # Model performance metrics
    accuracy = Column(Float, nullable=True)
    dice_score = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    
    # Model configuration
    input_size = Column(JSON, nullable=True)  # Expected input dimensions
    preprocessing_steps = Column(JSON, nullable=True)  # Required preprocessing
    postprocessing_steps = Column(JSON, nullable=True)  # Output processing
    
    # Deployment information
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Model artifacts
    model_path = Column(String(500), nullable=True)  # Path to model file
    config_path = Column(String(500), nullable=True)  # Path to config file


class UserFeedback(Base):
    """Model for collecting user feedback on processing results."""
    
    __tablename__ = "user_feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(UUID(as_uuid=True), nullable=False)
    user_id = Column(String(255), nullable=True)
    
    # Feedback content
    rating = Column(Integer, nullable=True)  # 1-5 scale
    feedback_text = Column(Text, nullable=True)
    is_accurate = Column(Boolean, nullable=True)  # For validation feedback
    
    # Feedback metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    feedback_type = Column(String(50), default="general")  # general, validation, bug_report


class SystemMetrics(Base):
    """Model for storing system performance metrics."""
    
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # API metrics
    total_requests = Column(Integer, default=0)
    successful_requests = Column(Integer, default=0)
    failed_requests = Column(Integer, default=0)
    average_response_time = Column(Float, default=0.0)
    
    # Model metrics
    model_inference_time = Column(Float, default=0.0)
    preprocessing_time = Column(Float, default=0.0)
    postprocessing_time = Column(Float, default=0.0)
    
    # Resource metrics
    cpu_usage = Column(Float, default=0.0)
    memory_usage = Column(Float, default=0.0)
    disk_usage = Column(Float, default=0.0)
    
    # Error tracking
    error_count = Column(Integer, default=0)
    error_types = Column(JSON, nullable=True)  # Count of different error types


class AuditLog(Base):
    """Model for audit logging and compliance tracking."""
    
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Event information
    event_type = Column(String(100), nullable=False)  # upload, process, download, delete
    user_id = Column(String(255), nullable=True)
    job_id = Column(UUID(as_uuid=True), nullable=True)
    
    # Event details
    description = Column(Text, nullable=False)
    ip_address = Column(String(45), nullable=True)  # IPv4 or IPv6
    user_agent = Column(String(500), nullable=True)
    
    # Data handling
    data_accessed = Column(JSON, nullable=True)  # What data was accessed
    anonymization_applied = Column(Boolean, default=True)
    
    # Compliance
    gdpr_compliant = Column(Boolean, default=True)
    hipaa_compliant = Column(Boolean, default=True)
