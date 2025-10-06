"""
Health check endpoint.
"""

import time
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ....core.database import get_db
from ....schemas.api import HealthResponse
from ....services.model_service import ModelService

router = APIRouter()

# Global start time for uptime calculation
start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db)) -> HealthResponse:
    """
    Health check endpoint.
    
    Returns the current status of the API and its dependencies.
    """
    try:
        # Check database connection
        db.execute("SELECT 1")
        db_status = "healthy"
    except Exception:
        db_status = "unhealthy"
    
    # Check model service
    model_service = ModelService()
    available_models = model_service.get_available_models()
    model_status = "healthy" if available_models else "no_models_loaded"
    
    # Calculate uptime
    uptime = time.time() - start_time
    
    # Determine overall status
    if db_status == "healthy" and model_status in ["healthy", "no_models_loaded"]:
        status = "healthy"
    else:
        status = "degraded"
    
    return HealthResponse(
        status=status,
        timestamp=datetime.utcnow(),
        version="1.0.0",
        uptime=uptime
    )


@router.get("/health/detailed")
async def detailed_health_check(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Detailed health check with component status.
    """
    health_info = {
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "uptime": time.time() - start_time,
        "components": {}
    }
    
    # Database health
    try:
        db.execute("SELECT 1")
        health_info["components"]["database"] = {
            "status": "healthy",
            "response_time_ms": 0  # Could measure actual response time
        }
    except Exception as e:
        health_info["components"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Model service health
    try:
        model_service = ModelService()
        available_models = model_service.get_available_models()
        health_info["components"]["model_service"] = {
            "status": "healthy",
            "loaded_models": len(available_models),
            "models": [model["model_id"] for model in available_models]
        }
    except Exception as e:
        health_info["components"]["model_service"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Redis health (if configured)
    try:
        from ....core.database import get_redis
        redis_client = get_redis()
        redis_client.ping()
        health_info["components"]["redis"] = {
            "status": "healthy"
        }
    except Exception as e:
        health_info["components"]["redis"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Determine overall status
    component_statuses = [comp["status"] for comp in health_info["components"].values()]
    if all(status == "healthy" for status in component_statuses):
        health_info["overall_status"] = "healthy"
    elif any(status == "unhealthy" for status in component_statuses):
        health_info["overall_status"] = "degraded"
    else:
        health_info["overall_status"] = "unknown"
    
    return health_info
