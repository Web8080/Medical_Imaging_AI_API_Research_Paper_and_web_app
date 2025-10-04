"""
Phase 5: Backend & API Design
Goal: Serve the model as an API endpoint for users with security and compliance.
"""

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import asyncio
from datetime import datetime

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn
from pydantic import BaseModel, Field
import jwt
from passlib.context import CryptContext
import redis
from sqlalchemy.orm import Session

# Medical imaging libraries
import pydicom
import nibabel as nib
from PIL import Image
import cv2

from ..core.database import get_db, engine
from ..core.security import verify_api_key, create_access_token, verify_token
from ..models.database import Job, Result, User
from ..schemas.api import (
    ProcessingRequest, ProcessingResponse, JobStatus, 
    ErrorResponse, HealthResponse, MetricsResponse
)
from ..services.model_service import ModelService
from ..services.dicom_processor import DICOMProcessor
from ..services.job_service import JobService
from ..services.visualization_service import VisualizationService
from ..evaluation.metrics import MedicalMetrics

logger = logging.getLogger(__name__)

# Security configuration
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Redis for caching
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)


class Phase5API:
    """
    Phase 5: Production-ready API with security and compliance features.
    
    Features:
    - Model packaging and serving
    - Inference pipeline
    - Security (HTTPS, JWT, API keys)
    - GDPR/HIPAA compliance
    - Performance monitoring
    - Rate limiting
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.app = FastAPI(
            title="Medical Imaging AI API",
            description="Advanced medical imaging AI for tumor detection and segmentation",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Initialize services
        self.model_service = ModelService(config)
        self.dicom_processor = DICOMProcessor()
        self.job_service = JobService()
        self.viz_service = VisualizationService()
        self.metrics = MedicalMetrics()
        
        # Setup middleware and routes
        self.setup_middleware()
        self.setup_routes()
        
        # Load models
        self.load_models()
    
    def setup_middleware(self):
        """Setup security and performance middleware."""
        
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.get('allowed_origins', ['*']),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Trusted host middleware
        self.app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=self.config.get('allowed_hosts', ['*'])
        )
        
        # Rate limiting middleware
        from slowapi import Limiter, _rate_limit_exceeded_handler
        from slowapi.util import get_remote_address
        from slowapi.errors import RateLimitExceeded
        
        limiter = Limiter(key_func=get_remote_address)
        self.app.state.limiter = limiter
        self.app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    def setup_routes(self):
        """Setup API routes."""
        
        # Health check
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            return HealthResponse(
                status="healthy",
                timestamp=datetime.utcnow(),
                version="1.0.0",
                models_loaded=len(self.model_service.loaded_models)
            )
        
        # Authentication endpoints
        @self.app.post("/auth/login")
        async def login(credentials: dict):
            """User login endpoint."""
            username = credentials.get('username')
            password = credentials.get('password')
            
            # Verify credentials (in production, use proper user database)
            if username == self.config.get('admin_username') and \
               pwd_context.verify(password, self.config.get('admin_password_hash')):
                
                access_token = create_access_token(data={"sub": username})
                return {"access_token": access_token, "token_type": "bearer"}
            else:
                raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Model upload endpoint
        @self.app.post("/upload", response_model=ProcessingResponse)
        @limiter.limit("10/minute")  # Rate limiting
        async def upload_medical_image(
            background_tasks: BackgroundTasks,
            file: UploadFile = File(...),
            model_type: str = "attention_unet",
            task_type: str = "segmentation",
            api_key: str = None,
            db: Session = Depends(get_db)
        ):
            """
            Upload medical image for processing.
            
            Args:
                file: Medical image file (DICOM, NIfTI, or standard image)
                model_type: Type of model to use
                task_type: Task type (segmentation, classification, detection)
                api_key: API key for authentication
            """
            try:
                # Verify API key
                if not verify_api_key(api_key):
                    raise HTTPException(status_code=401, detail="Invalid API key")
                
                # Validate file
                if not self._validate_file(file):
                    raise HTTPException(status_code=400, detail="Invalid file format")
                
                # Create job
                job_id = str(uuid.uuid4())
                job = self.job_service.create_job(
                    db, job_id, model_type, task_type, file.filename
                )
                
                # Save uploaded file
                file_path = self._save_uploaded_file(file, job_id)
                
                # Start background processing
                background_tasks.add_task(
                    self._process_image_background,
                    job_id, file_path, model_type, task_type
                )
                
                return ProcessingResponse(
                    job_id=job_id,
                    status="processing",
                    message="Image uploaded successfully. Processing started.",
                    estimated_time=self._estimate_processing_time(model_type, task_type)
                )
                
            except Exception as e:
                logger.error(f"Error in upload endpoint: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Get result endpoint
        @self.app.get("/result/{job_id}", response_model=JobStatus)
        async def get_result(job_id: str, db: Session = Depends(get_db)):
            """Get processing result for a job."""
            try:
                job = self.job_service.get_job_by_id(db, job_id)
                if not job:
                    raise HTTPException(status_code=404, detail="Job not found")
                
                return JobStatus(
                    job_id=job_id,
                    status=job.status,
                    created_at=job.created_at,
                    updated_at=job.updated_at,
                    results=job.results,
                    error_message=job.error_message
                )
                
            except Exception as e:
                logger.error(f"Error getting result: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Metrics endpoint
        @self.app.get("/metrics", response_model=MetricsResponse)
        async def get_metrics(api_key: str = None):
            """Get API performance metrics."""
            try:
                # Verify API key
                if not verify_api_key(api_key):
                    raise HTTPException(status_code=401, detail="Invalid API key")
                
                metrics = self._get_performance_metrics()
                return MetricsResponse(**metrics)
                
            except Exception as e:
                logger.error(f"Error getting metrics: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Feedback endpoint
        @self.app.post("/feedback")
        async def submit_feedback(
            job_id: str,
            feedback: dict,
            rating: int = Field(..., ge=1, le=5),
            db: Session = Depends(get_db)
        ):
            """Submit user feedback for a job."""
            try:
                # Update job with feedback
                job = self.job_service.get_job_by_id(db, job_id)
                if not job:
                    raise HTTPException(status_code=404, detail="Job not found")
                
                job.feedback = feedback
                job.rating = rating
                job.updated_at = datetime.utcnow()
                
                db.commit()
                
                return {"message": "Feedback submitted successfully"}
                
            except Exception as e:
                logger.error(f"Error submitting feedback: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Model management endpoints
        @self.app.get("/models")
        async def list_models(api_key: str = None):
            """List available models."""
            try:
                if not verify_api_key(api_key):
                    raise HTTPException(status_code=401, detail="Invalid API key")
                
                models = self.model_service.list_available_models()
                return {"models": models}
                
            except Exception as e:
                logger.error(f"Error listing models: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/models/{model_name}/load")
        async def load_model(model_name: str, api_key: str = None):
            """Load a specific model."""
            try:
                if not verify_api_key(api_key):
                    raise HTTPException(status_code=401, detail="Invalid API key")
                
                success = self.model_service.load_model(model_name)
                if success:
                    return {"message": f"Model {model_name} loaded successfully"}
                else:
                    raise HTTPException(status_code=400, detail="Failed to load model")
                
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Visualization endpoints
        @self.app.get("/visualizations/{job_id}/confusion-matrix")
        async def get_confusion_matrix(job_id: str, db: Session = Depends(get_db)):
            """Get confusion matrix visualization for a job."""
            try:
                job = self.job_service.get_job_by_id(db, job_id)
                if not job:
                    raise HTTPException(status_code=404, detail="Job not found")
                
                if not job.results:
                    raise HTTPException(status_code=404, detail="No results found")
                
                # Generate confusion matrix
                y_true = [r.true_label for r in job.results if r.true_label is not None]
                y_pred = [r.predicted_label for r in job.results if r.predicted_label is not None]
                
                if not y_true or not y_pred:
                    raise HTTPException(status_code=400, detail="Insufficient data")
                
                cm_base64 = self.viz_service.generate_confusion_matrix(y_true, y_pred, ['Normal', 'Abnormal'])
                
                return {"image_base64": cm_base64}
                
            except Exception as e:
                logger.error(f"Error generating confusion matrix: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def load_models(self):
        """Load models for serving."""
        try:
            models_to_load = self.config.get('models_to_load', ['attention_unet', 'vit_unet'])
            
            for model_name in models_to_load:
                success = self.model_service.load_model(model_name)
                if success:
                    logger.info(f"Model {model_name} loaded successfully")
                else:
                    logger.warning(f"Failed to load model {model_name}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def _validate_file(self, file: UploadFile) -> bool:
        """Validate uploaded file."""
        allowed_extensions = {'.dcm', '.dicom', '.nii', '.nii.gz', '.jpg', '.jpeg', '.png', '.tiff'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            return False
        
        # Check file size (max 100MB)
        if file.size and file.size > 100 * 1024 * 1024:
            return False
        
        return True
    
    def _save_uploaded_file(self, file: UploadFile, job_id: str) -> Path:
        """Save uploaded file to storage."""
        upload_dir = Path(self.config.get('upload_dir', 'uploads'))
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / f"{job_id}_{file.filename}"
        
        with open(file_path, "wb") as buffer:
            content = file.file.read()
            buffer.write(content)
        
        return file_path
    
    def _estimate_processing_time(self, model_type: str, task_type: str) -> int:
        """Estimate processing time in seconds."""
        base_time = 30  # Base processing time
        
        # Adjust based on model complexity
        if model_type == 'ensemble':
            base_time *= 3
        elif model_type == 'vit_unet':
            base_time *= 2
        elif model_type == 'efficientnet_unet':
            base_time *= 1.5
        
        # Adjust based on task type
        if task_type == 'detection':
            base_time *= 1.5
        elif task_type == 'classification':
            base_time *= 0.5
        
        return base_time
    
    async def _process_image_background(self, job_id: str, file_path: Path, 
                                      model_type: str, task_type: str):
        """Background task for processing images."""
        try:
            # Update job status
            db = next(get_db())
            job = self.job_service.get_job_by_id(db, job_id)
            job.status = "processing"
            job.updated_at = datetime.utcnow()
            db.commit()
            
            # Load and preprocess image
            image_data = self._load_image(file_path)
            processed_image = self._preprocess_image(image_data)
            
            # Run inference
            results = self.model_service.run_inference(
                processed_image, model_type, task_type
            )
            
            # Post-process results
            processed_results = self._postprocess_results(results, task_type)
            
            # Save results
            self.job_service.save_results(db, job_id, processed_results)
            
            # Update job status
            job.status = "completed"
            job.updated_at = datetime.utcnow()
            db.commit()
            
            logger.info(f"Job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Error processing job {job_id}: {e}")
            
            # Update job with error
            db = next(get_db())
            job = self.job_service.get_job_by_id(db, job_id)
            job.status = "failed"
            job.error_message = str(e)
            job.updated_at = datetime.utcnow()
            db.commit()
    
    def _load_image(self, file_path: Path) -> np.ndarray:
        """Load medical image from file."""
        file_extension = file_path.suffix.lower()
        
        if file_extension in ['.dcm', '.dicom']:
            # Load DICOM
            ds = pydicom.dcmread(file_path)
            return ds.pixel_array.astype(np.float32)
        
        elif file_extension in ['.nii', '.gz']:
            # Load NIfTI
            nii = nib.load(file_path)
            return nii.get_fdata().astype(np.float32)
        
        else:
            # Load standard image
            image = Image.open(file_path)
            return np.array(image).astype(np.float32)
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model inference."""
        # Normalize
        image = (image - np.mean(image)) / (np.std(image) + 1e-8)
        
        # Resize if needed
        if len(image.shape) == 2:
            image = cv2.resize(image, (224, 224))
            image = np.expand_dims(image, axis=0)  # Add channel dimension
        elif len(image.shape) == 3:
            image = cv2.resize(image, (224, 224, image.shape[2]))
            image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        
        # Convert to tensor
        tensor = torch.FloatTensor(image).unsqueeze(0)  # Add batch dimension
        
        return tensor
    
    def _postprocess_results(self, results: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """Post-process model results."""
        processed_results = {}
        
        if task_type == 'segmentation':
            # Convert segmentation mask to binary
            mask = results.get('segmentation', results.get('output'))
            if mask is not None:
                binary_mask = (mask > 0.5).astype(np.uint8)
                processed_results['segmentation_mask'] = binary_mask.tolist()
                
                # Calculate volume metrics
                volume = np.sum(binary_mask)
                processed_results['volume_voxels'] = int(volume)
                processed_results['volume_mm3'] = float(volume * 1.0)  # Assuming 1mm³ voxels
        
        elif task_type == 'classification':
            # Get classification probabilities
            probs = results.get('classification', results.get('output'))
            if probs is not None:
                if isinstance(probs, torch.Tensor):
                    probs = torch.softmax(probs, dim=1).numpy()
                
                predicted_class = int(np.argmax(probs))
                confidence = float(np.max(probs))
                
                processed_results['predicted_class'] = predicted_class
                processed_results['confidence'] = confidence
                processed_results['class_probabilities'] = probs.tolist()
        
        elif task_type == 'detection':
            # Process detection results
            detections = results.get('detections', results.get('output'))
            if detections is not None:
                processed_results['detections'] = detections
                processed_results['num_detections'] = len(detections)
        
        return processed_results
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get API performance metrics."""
        try:
            # Get metrics from Redis cache
            total_requests = redis_client.get('total_requests') or 0
            successful_requests = redis_client.get('successful_requests') or 0
            failed_requests = redis_client.get('failed_requests') or 0
            avg_processing_time = redis_client.get('avg_processing_time') or 0
            
            return {
                'total_requests': int(total_requests),
                'successful_requests': int(successful_requests),
                'failed_requests': int(failed_requests),
                'success_rate': float(successful_requests) / max(int(total_requests), 1),
                'avg_processing_time_seconds': float(avg_processing_time),
                'models_loaded': len(self.model_service.loaded_models),
                'uptime_seconds': time.time() - self.start_time
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'success_rate': 0.0,
                'avg_processing_time_seconds': 0.0,
                'models_loaded': len(self.model_service.loaded_models),
                'uptime_seconds': 0
            }
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the API server."""
        self.start_time = time.time()
        
        logger.info(f"Starting Medical Imaging AI API on {host}:{port}")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info",
            ssl_keyfile=self.config.get('ssl_keyfile'),
            ssl_certfile=self.config.get('ssl_certfile')
        )


def create_api_app(config: Dict[str, Any]) -> FastAPI:
    """Create and configure the API application."""
    api = Phase5API(config)
    return api.app


def main():
    """Main function to run the API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Medical Imaging AI API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--config", type=Path, help="Configuration file")
    parser.add_argument("--ssl", action="store_true", help="Enable SSL")
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config and args.config.exists():
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Add SSL configuration
    if args.ssl:
        config['ssl_keyfile'] = 'ssl/private.key'
        config['ssl_certfile'] = 'ssl/certificate.crt'
    
    # Create and run API
    api = Phase5API(config)
    api.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
