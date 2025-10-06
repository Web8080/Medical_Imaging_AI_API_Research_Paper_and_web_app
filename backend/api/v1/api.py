"""
API v1 router configuration.
"""

from fastapi import APIRouter

from .endpoints import health, upload, visualizations

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(health.router, tags=["health"])
api_router.include_router(upload.router, prefix="/upload", tags=["upload"])
api_router.include_router(visualizations.router, prefix="/visualizations", tags=["visualizations"])
