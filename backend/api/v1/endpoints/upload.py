"""
File upload and processing endpoints.
"""

import asyncio
import logging
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, BackgroundTasks
from sqlalchemy.orm import Session

from ....core.database import get_db, get_redis
from ....core.security import verify_token
from ....models.database import ProcessingJob
from ....schemas.api import UploadResponse, ProcessingStatus, ErrorResponse
from ....services.dicom_processor import DICOMProcessor
from ....services.model_service import ModelService
from ....services.job_service import JobService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    redis_client = Depends(get_redis)
) -> UploadResponse:
    """
    Upload a medical imaging file for processing.
    
    Args:
        file: Medical imaging file (DICOM, NIfTI, JPEG, PNG)
        model_id: Optional model ID to use for processing
        user_id: Optional user identifier
        background_tasks: FastAPI background tasks
        db: Database session
        redis_client: Redis client
        
    Returns:
        Upload response with job ID
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Read file content
        file_content = await file.read()
        
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Create job record
        job_id = uuid4()
        job = ProcessingJob(
            id=job_id,
            user_id=user_id,
            status="pending",
            original_filename=file.filename,
            file_size=len(file_content),
            file_format="unknown",  # Will be updated during processing
            file_hash="",  # Will be calculated during processing
            anonymized=True
        )
        
        db.add(job)
        db.commit()
        db.refresh(job)
        
        # Start background processing
        background_tasks.add_task(
            process_file_background,
            job_id,
            file_content,
            file.filename,
            model_id,
            user_id
        )
        
        # Store job status in Redis for quick access
        job_service = JobService()
        job_service.update_job_status(redis_client, str(job_id), "pending")
        
        logger.info(f"File upload initiated for job {job_id}")
        
        return UploadResponse(
            job_id=job_id,
            status="pending",
            message="File uploaded successfully",
            estimated_processing_time=30  # Estimated 30 seconds
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/jobs/{job_id}", response_model=ProcessingStatus)
async def get_job_status(
    job_id: str,
    db: Session = Depends(get_db),
    redis_client = Depends(get_redis)
) -> ProcessingStatus:
    """
    Get the status of a processing job.
    
    Args:
        job_id: Job identifier
        db: Database session
        redis_client: Redis client
        
    Returns:
        Job status information
    """
    try:
        job_service = JobService()
        job_status = job_service.get_job_status(db, redis_client, job_id)
        
        if not job_status:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return job_status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving job status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve job status: {str(e)}")


@router.get("/jobs", response_model=List[ProcessingStatus])
async def list_jobs(
    user_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db)
) -> List[ProcessingStatus]:
    """
    List processing jobs with optional filtering.
    
    Args:
        user_id: Filter by user ID
        status: Filter by status
        limit: Maximum number of jobs to return
        offset: Number of jobs to skip
        db: Database session
        
    Returns:
        List of job statuses
    """
    try:
        job_service = JobService()
        jobs = job_service.list_jobs(db, user_id, status, limit, offset)
        
        return jobs
        
    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")


async def process_file_background(
    job_id: str,
    file_content: bytes,
    filename: str,
    model_id: Optional[str],
    user_id: Optional[str]
):
    """
    Background task for processing uploaded files.
    
    Args:
        job_id: Job identifier
        file_content: File content
        filename: Original filename
        model_id: Model ID to use
        user_id: User identifier
    """
    from ....core.database import SessionLocal
    
    db = SessionLocal()
    redis_client = get_redis()
    
    try:
        # Update job status to processing
        job_service = JobService()
        job_service.update_job_status(redis_client, job_id, "processing")
        
        # Process the file
        dicom_processor = DICOMProcessor()
        processed_data = dicom_processor.process_upload(file_content, filename)
        
        # Update job with file information
        job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
        if job:
            job.file_format = processed_data["format"]
            job.file_hash = processed_data["file_hash"]
            db.commit()
        
        # Run model inference if model is specified
        if model_id:
            model_service = ModelService()
            
            # Check if model is loaded
            available_models = model_service.get_available_models()
            if not any(model["model_id"] == model_id for model in available_models):
                # Load default model configuration
                model_config = {
                    "type": "pytorch",
                    "task_type": "segmentation",
                    "input_size": [256, 256],
                    "threshold": 0.5
                }
                # In a real implementation, you would load the actual model
                # model_service.load_model(model_id, model_path, model_config)
            
            # Preprocess data for model
            model_requirements = {
                "input_size": [256, 256],
                "augmentation": False
            }
            preprocessed_data = dicom_processor.preprocess_for_model(
                processed_data, model_requirements
            )
            
            # Run inference
            inference_results = model_service.predict(
                model_id, preprocessed_data, model_requirements
            )
            
            # Update job with results
            if job:
                job.status = "completed"
                job.model_used = model_id
                job.processing_time_seconds = inference_results["inference_time"]
                job.results = {
                    "detections": [detection.dict() for detection in inference_results["predictions"]],
                    "summary": {
                        "total_detections": len(inference_results["predictions"]),
                        "average_confidence": sum(d.class_confidence for d in inference_results["predictions"]) / max(1, len(inference_results["predictions"]))
                    }
                }
                job.confidence_scores = {
                    "overall_confidence": inference_results["predictions"][0].class_confidence if inference_results["predictions"] else 0.0
                }
                db.commit()
        else:
            # No model specified, just mark as completed
            if job:
                job.status = "completed"
                job.processing_time_seconds = 0.0
                job.results = {
                    "message": "File processed successfully, no model inference performed"
                }
                db.commit()
        
        # Update Redis status
        job_service.update_job_status(redis_client, job_id, "completed")
        
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {e}")
        
        # Update job with error
        job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
        if job:
            job.status = "failed"
            job.error_message = str(e)
            db.commit()
        
        # Update Redis status
        job_service.update_job_status(redis_client, job_id, "failed")
        
    finally:
        db.close()
