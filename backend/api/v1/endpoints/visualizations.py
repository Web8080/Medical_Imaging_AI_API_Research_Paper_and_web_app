"""
Visualization endpoints for model evaluation and analysis.
"""

import logging
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ....core.database import get_db
from ....schemas.api import ProcessingResult, ErrorResponse
from ....services.visualization_service import VisualizationService
from ....services.job_service import JobService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/visualizations/types")
async def get_visualization_types():
    """
    Get list of available visualization types.
    
    Returns:
        List of available visualization types with descriptions
    """
    try:
        viz_service = VisualizationService()
        return viz_service.get_visualization_types()
    except Exception as e:
        logger.error(f"Error getting visualization types: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get visualization types: {str(e)}")


@router.get("/results/{job_id}/visualizations")
async def get_result_visualizations(
    job_id: str,
    viz_type: Optional[str] = Query(None, description="Type of visualization to generate"),
    db: Session = Depends(get_db)
) -> Dict[str, str]:
    """
    Generate visualization plots for processing results.
    
    Args:
        job_id: Job identifier
        viz_type: Optional specific visualization type
        db: Database session
        
    Returns:
        Dictionary containing base64-encoded visualization images
    """
    try:
        # Get job results
        job_service = JobService()
        job_results = job_service.get_job_results(db, job_id)
        
        if not job_results:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job_results["status"] != "completed":
            raise HTTPException(status_code=400, detail="Job not completed yet")
        
        # Create visualization service
        viz_service = VisualizationService()
        
        # Convert job results to ProcessingResult object
        # This is a simplified conversion - in practice, you'd have proper serialization
        result_data = {
            "job_id": job_results["job_id"],
            "status": job_results["status"],
            "detections": job_results.get("results", {}).get("detections", []),
            "model_used": job_results.get("model_used", "unknown"),
            "processing_time_seconds": job_results.get("processing_time_seconds", 0),
            "overall_confidence": job_results.get("confidence_scores", {}).get("overall_confidence", 0.0)
        }
        
        # Generate visualizations
        if viz_type:
            # Generate specific visualization type
            visualizations = {}
            if viz_type == "confidence_distribution":
                # Extract confidence scores from detections
                detections = result_data["detections"]
                if detections:
                    confidences = [det.get("class_confidence", 0.0) for det in detections]
                    labels = [1] * len(confidences)  # All are positive detections
                    visualizations[viz_type] = viz_service.create_detection_confidence_distribution(
                        confidences, labels
                    )
            elif viz_type == "volume_distribution":
                # Extract volume measurements
                detections = result_data["detections"]
                volumes = []
                for det in detections:
                    if det.get("metrics") and det["metrics"].get("volume_mm3"):
                        volumes.append(det["metrics"]["volume_mm3"])
                
                if volumes:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.hist(volumes, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
                    ax.set_xlabel('Volume (mm³)')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Tumor Volume Distribution')
                    ax.grid(True, alpha=0.3)
                    visualizations[viz_type] = viz_service._fig_to_base64(fig)
        else:
            # Generate all available visualizations
            visualizations = viz_service.create_processing_result_summary(result_data)
        
        return visualizations
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate visualizations: {str(e)}")


@router.post("/evaluation/dashboard")
async def create_evaluation_dashboard(
    evaluation_data: Dict[str, any],
    db: Session = Depends(get_db)
) -> Dict[str, str]:
    """
    Create comprehensive evaluation dashboard.
    
    Args:
        evaluation_data: Dictionary containing evaluation metrics and data
        db: Database session
        
    Returns:
        Dictionary containing base64-encoded dashboard visualizations
    """
    try:
        viz_service = VisualizationService()
        dashboard = viz_service.create_evaluation_dashboard(evaluation_data)
        
        return dashboard
        
    except Exception as e:
        logger.error(f"Error creating evaluation dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create dashboard: {str(e)}")


@router.get("/models/{model_id}/performance")
async def get_model_performance_visualizations(
    model_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, str]:
    """
    Get performance visualizations for a specific model.
    
    Args:
        model_id: Model identifier
        db: Database session
        
    Returns:
        Dictionary containing model performance visualizations
    """
    try:
        # This would typically query the database for model performance data
        # For now, we'll return a placeholder response
        
        viz_service = VisualizationService()
        
        # Example performance data (in practice, this would come from the database)
        example_data = {
            "y_true": [1, 0, 1, 1, 0, 1, 0, 0, 1, 1],
            "y_pred": [1, 0, 1, 0, 0, 1, 1, 0, 1, 1],
            "y_scores": [0.9, 0.1, 0.8, 0.4, 0.2, 0.95, 0.6, 0.1, 0.85, 0.9],
            "class_names": ["Normal", "Abnormal"],
            "model_name": model_id,
            "dice_scores": [0.85, 0.92, 0.78, 0.88, 0.91, 0.83, 0.89, 0.87, 0.90, 0.86]
        }
        
        dashboard = viz_service.create_evaluation_dashboard(example_data)
        
        return dashboard
        
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model performance: {str(e)}")


@router.get("/comparison/models")
async def compare_models_visualizations(
    model_ids: List[str] = Query(..., description="List of model IDs to compare"),
    db: Session = Depends(get_db)
) -> Dict[str, str]:
    """
    Create visualizations comparing multiple models.
    
    Args:
        model_ids: List of model identifiers to compare
        db: Database session
        
    Returns:
        Dictionary containing model comparison visualizations
    """
    try:
        viz_service = VisualizationService()
        
        # Example comparison data (in practice, this would come from the database)
        models_metrics = {}
        for model_id in model_ids:
            models_metrics[model_id] = {
                "accuracy": 0.85 + (hash(model_id) % 100) / 1000,  # Simulated metrics
                "precision": 0.82 + (hash(model_id) % 100) / 1000,
                "recall": 0.88 + (hash(model_id) % 100) / 1000,
                "f1_score": 0.85 + (hash(model_id) % 100) / 1000,
                "dice_score": 0.87 + (hash(model_id) % 100) / 1000
            }
        
        comparison_data = {
            "models_metrics": models_metrics,
            "metrics": ["accuracy", "precision", "recall", "f1_score", "dice_score"]
        }
        
        visualizations = {}
        visualizations["metrics_comparison"] = viz_service.create_metrics_comparison(
            models_metrics, 
            ["accuracy", "precision", "recall", "f1_score"]
        )
        
        return visualizations
        
    except Exception as e:
        logger.error(f"Error comparing models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to compare models: {str(e)}")


@router.get("/statistics/summary")
async def get_statistics_summary(
    db: Session = Depends(get_db)
) -> Dict[str, any]:
    """
    Get statistical summary of all processing jobs.
    
    Args:
        db: Database session
        
    Returns:
        Dictionary containing statistical summaries and visualizations
    """
    try:
        # This would typically query the database for comprehensive statistics
        # For now, we'll return example data
        
        summary = {
            "total_jobs": 1250,
            "completed_jobs": 1180,
            "failed_jobs": 70,
            "average_processing_time": 2.3,
            "total_detections": 3450,
            "average_confidence": 0.87,
            "models_used": {
                "brain_segmentation": 450,
                "lung_detection": 380,
                "liver_segmentation": 350
            },
            "file_formats": {
                "DICOM": 800,
                "NIfTI": 300,
                "JPEG": 100,
                "PNG": 50
            }
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Error getting statistics summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")
