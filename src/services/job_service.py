"""
Job management service for processing jobs.
"""

import logging
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import desc

from ..models.database import ProcessingJob
from ..schemas.api import ProcessingStatus

logger = logging.getLogger(__name__)


class JobService:
    """Service for managing processing jobs."""
    
    def get_job_status(self, db: Session, redis_client, job_id: str) -> Optional[ProcessingStatus]:
        """
        Get job status from database and cache.
        
        Args:
            db: Database session
            redis_client: Redis client
            job_id: Job identifier
            
        Returns:
            Job status or None if not found
        """
        try:
            # Try to get from Redis cache first
            cached_status = redis_client.get(f"job_status:{job_id}")
            if cached_status:
                # Parse cached data and return
                import json
                cached_data = json.loads(cached_status)
                return ProcessingStatus(**cached_data)
            
            # Get from database
            job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
            if not job:
                return None
            
            # Create status object
            status = ProcessingStatus(
                job_id=job.id,
                status=job.status,
                created_at=job.created_at,
                updated_at=job.updated_at,
                completed_at=job.completed_at,
                error_message=job.error_message
            )
            
            # Cache the result
            self.update_job_status(redis_client, job_id, job.status, status.dict())
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting job status: {e}")
            return None
    
    def update_job_status(self, redis_client, job_id: str, status: str, full_status: Optional[dict] = None):
        """
        Update job status in Redis cache.
        
        Args:
            redis_client: Redis client
            job_id: Job identifier
            status: New status
            full_status: Optional full status object to cache
        """
        try:
            if full_status:
                import json
                redis_client.setex(
                    f"job_status:{job_id}",
                    3600,  # Cache for 1 hour
                    json.dumps(full_status, default=str)
                )
            else:
                redis_client.setex(f"job_status:{job_id}", 3600, status)
                
        except Exception as e:
            logger.error(f"Error updating job status in cache: {e}")
    
    def list_jobs(
        self,
        db: Session,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[ProcessingStatus]:
        """
        List jobs with optional filtering.
        
        Args:
            db: Database session
            user_id: Filter by user ID
            status: Filter by status
            limit: Maximum number of jobs
            offset: Number of jobs to skip
            
        Returns:
            List of job statuses
        """
        try:
            query = db.query(ProcessingJob)
            
            # Apply filters
            if user_id:
                query = query.filter(ProcessingJob.user_id == user_id)
            if status:
                query = query.filter(ProcessingJob.status == status)
            
            # Order by creation time (newest first)
            query = query.order_by(desc(ProcessingJob.created_at))
            
            # Apply pagination
            jobs = query.offset(offset).limit(limit).all()
            
            # Convert to status objects
            statuses = []
            for job in jobs:
                status_obj = ProcessingStatus(
                    job_id=job.id,
                    status=job.status,
                    created_at=job.created_at,
                    updated_at=job.updated_at,
                    completed_at=job.completed_at,
                    error_message=job.error_message
                )
                statuses.append(status_obj)
            
            return statuses
            
        except Exception as e:
            logger.error(f"Error listing jobs: {e}")
            return []
    
    def get_job_results(self, db: Session, job_id: str) -> Optional[dict]:
        """
        Get job results.
        
        Args:
            db: Database session
            job_id: Job identifier
            
        Returns:
            Job results or None if not found
        """
        try:
            job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
            if not job:
                return None
            
            return {
                "job_id": str(job.id),
                "status": job.status,
                "results": job.results,
                "confidence_scores": job.confidence_scores,
                "model_used": job.model_used,
                "processing_time_seconds": job.processing_time_seconds,
                "created_at": job.created_at,
                "completed_at": job.completed_at,
                "error_message": job.error_message
            }
            
        except Exception as e:
            logger.error(f"Error getting job results: {e}")
            return None
    
    def delete_job(self, db: Session, job_id: str, user_id: Optional[str] = None) -> bool:
        """
        Delete a job (for GDPR compliance).
        
        Args:
            db: Database session
            job_id: Job identifier
            user_id: User identifier (for authorization)
            
        Returns:
            True if deleted successfully
        """
        try:
            query = db.query(ProcessingJob).filter(ProcessingJob.id == job_id)
            
            # Add user authorization if provided
            if user_id:
                query = query.filter(ProcessingJob.user_id == user_id)
            
            job = query.first()
            if not job:
                return False
            
            db.delete(job)
            db.commit()
            
            logger.info(f"Job {job_id} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting job: {e}")
            db.rollback()
            return False
    
    def cleanup_old_jobs(self, db: Session, days_old: int = 30) -> int:
        """
        Clean up old completed jobs for data retention compliance.
        
        Args:
            db: Database session
            days_old: Number of days after which to delete jobs
            
        Returns:
            Number of jobs deleted
        """
        try:
            from datetime import timedelta
            
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            # Find old completed jobs
            old_jobs = db.query(ProcessingJob).filter(
                ProcessingJob.status == "completed",
                ProcessingJob.completed_at < cutoff_date
            ).all()
            
            # Delete them
            count = len(old_jobs)
            for job in old_jobs:
                db.delete(job)
            
            db.commit()
            
            logger.info(f"Cleaned up {count} old jobs")
            return count
            
        except Exception as e:
            logger.error(f"Error cleaning up old jobs: {e}")
            db.rollback()
            return 0
