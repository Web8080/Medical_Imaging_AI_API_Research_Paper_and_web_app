#!/usr/bin/env python3
"""
Medical Imaging AI API - Main FastAPI Application
Phase 5: Backend & API Design

This module implements the main FastAPI application for the Medical Imaging AI API,
following the specifications outlined in the research paper and roadmap.
"""

import os
import uuid
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import json
import hashlib
import secrets

# Import our trained models
from models.model_loader import ModelLoader
from preprocessing.image_processor import ImageProcessor
from utils.security import SecurityManager
from utils.database import DatabaseManager
from utils.metrics import MetricsCalculator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Medical Imaging AI API",
    description="AI-powered medical imaging analysis for tumor detection and measurement",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
security_manager = SecurityManager()

# Initialize components
model_loader = ModelLoader()
image_processor = ImageProcessor()
db_manager = DatabaseManager()
metrics_calculator = MetricsCalculator()

# Pydantic models for request/response
class AnalysisRequest(BaseModel):
    """Request model for image analysis."""
    model_type: str = Field(..., description="Type of model to use (chest, derma, oct)")
    analysis_type: str = Field(..., description="Type of analysis (classification, segmentation)")
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Confidence threshold for predictions")

class AnalysisResponse(BaseModel):
    """Response model for image analysis."""
    request_id: str
    status: str
    results: Dict[str, Any]
    confidence: float
    processing_time: float
    timestamp: datetime
    model_used: str

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    models_loaded: List[str]
    system_info: Dict[str, Any]

class MetricsResponse(BaseModel):
    """Metrics response model."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_processing_time: float
    model_performance: Dict[str, Any]

# Dependency for authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token."""
    if not security_manager.verify_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid API token")
    return credentials.credentials

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting Medical Imaging AI API...")
    
    # Load models
    await model_loader.load_all_models()
    logger.info("All models loaded successfully")
    
    # Initialize database
    await db_manager.initialize()
    logger.info("Database initialized")
    
    logger.info("Medical Imaging AI API started successfully")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        models_loaded=model_loader.get_loaded_models(),
        system_info={
            "python_version": "3.9+",
            "pytorch_version": torch.__version__,
            "gpu_available": torch.cuda.is_available(),
            "memory_usage": "N/A"  # Add memory monitoring
        }
    )

# Main analysis endpoint
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_type: str = "chest",
    analysis_type: str = "classification",
    confidence_threshold: float = 0.5,
    token: str = Depends(verify_token)
):
    """
    Analyze a medical image using AI models.
    
    Args:
        file: Medical image file (DICOM, NIfTI, or standard image formats)
        model_type: Type of model to use (chest, derma, oct)
        analysis_type: Type of analysis (classification, segmentation)
        confidence_threshold: Confidence threshold for predictions
        token: API authentication token
    
    Returns:
        Analysis results with predictions and confidence scores
    """
    start_time = datetime.utcnow()
    request_id = str(uuid.uuid4())
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Read file content
        file_content = await file.read()
        
        # Generate file hash for deduplication
        file_hash = hashlib.sha256(file_content).hexdigest()
        
        # Check if we've processed this file before
        cached_result = await db_manager.get_cached_result(file_hash, model_type)
        if cached_result:
            logger.info(f"Returning cached result for {request_id}")
            return AnalysisResponse(**cached_result)
        
        # Process image
        processed_image = await image_processor.process_image(file_content, file.filename)
        
        # Load appropriate model
        model = model_loader.get_model(model_type)
        if not model:
            raise HTTPException(status_code=400, detail=f"Model {model_type} not available")
        
        # Perform analysis
        if analysis_type == "classification":
            results = await perform_classification(model, processed_image, model_type)
        elif analysis_type == "segmentation":
            results = await perform_segmentation(model, processed_image, model_type)
        else:
            raise HTTPException(status_code=400, detail="Invalid analysis type")
        
        # Calculate confidence
        confidence = calculate_confidence(results)
        
        # Filter results by confidence threshold
        filtered_results = filter_by_confidence(results, confidence_threshold)
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Create response
        response = AnalysisResponse(
            request_id=request_id,
            status="success",
            results=filtered_results,
            confidence=confidence,
            processing_time=processing_time,
            timestamp=datetime.utcnow(),
            model_used=model_type
        )
        
        # Cache result
        background_tasks.add_task(
            db_manager.cache_result, 
            file_hash, 
            model_type, 
            response.dict()
        )
        
        # Log metrics
        background_tasks.add_task(
            metrics_calculator.log_request,
            request_id,
            model_type,
            processing_time,
            True
        )
        
        logger.info(f"Analysis completed for {request_id} in {processing_time:.2f}s")
        return response
        
    except Exception as e:
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        logger.error(f"Analysis failed for {request_id}: {str(e)}")
        
        # Log failed request
        background_tasks.add_task(
            metrics_calculator.log_request,
            request_id,
            model_type,
            processing_time,
            False
        )
        
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Get analysis result by ID
@app.get("/result/{request_id}", response_model=AnalysisResponse)
async def get_result(request_id: str, token: str = Depends(verify_token)):
    """Get analysis result by request ID."""
    result = await db_manager.get_result(request_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    return AnalysisResponse(**result)

# Get system metrics
@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics(token: str = Depends(verify_token)):
    """Get system performance metrics."""
    metrics = await metrics_calculator.get_metrics()
    return MetricsResponse(**metrics)

# Feedback endpoint
@app.post("/feedback")
async def submit_feedback(
    request_id: str,
    feedback: Dict[str, Any],
    token: str = Depends(verify_token)
):
    """Submit feedback for analysis results."""
    await db_manager.store_feedback(request_id, feedback)
    return {"status": "feedback_received", "request_id": request_id}

# Model management endpoints
@app.get("/models")
async def list_models(token: str = Depends(verify_token)):
    """List available models."""
    return {
        "available_models": model_loader.get_loaded_models(),
        "model_info": model_loader.get_model_info()
    }

@app.post("/models/{model_type}/reload")
async def reload_model(model_type: str, token: str = Depends(verify_token)):
    """Reload a specific model."""
    success = await model_loader.reload_model(model_type)
    if success:
        return {"status": "model_reloaded", "model_type": model_type}
    else:
        raise HTTPException(status_code=400, detail="Failed to reload model")

# Utility functions
async def perform_classification(model, image, model_type):
    """Perform image classification."""
    with torch.no_grad():
        # Preprocess image for model
        input_tensor = image_processor.prepare_for_model(image, model_type)
        
        # Get prediction
        outputs = model(input_tensor)
        
        # Apply softmax for probabilities
        probabilities = F.softmax(outputs, dim=1)
        
        # Get class predictions
        _, predicted_classes = torch.max(probabilities, 1)
        
        # Get class names
        class_names = model_loader.get_class_names(model_type)
        
        # Format results
        results = {
            "predictions": [],
            "top_prediction": {
                "class": class_names[predicted_classes[0].item()],
                "confidence": probabilities[0][predicted_classes[0]].item()
            }
        }
        
        # Add all predictions
        for i, prob in enumerate(probabilities[0]):
            results["predictions"].append({
                "class": class_names[i],
                "confidence": prob.item()
            })
        
        return results

async def perform_segmentation(model, image, model_type):
    """Perform image segmentation."""
    with torch.no_grad():
        # Preprocess image for model
        input_tensor = image_processor.prepare_for_model(image, model_type)
        
        # Get segmentation mask
        outputs = model(input_tensor)
        
        # Apply sigmoid for binary segmentation
        if outputs.shape[1] == 1:
            mask = torch.sigmoid(outputs)
        else:
            mask = F.softmax(outputs, dim=1)
        
        # Convert to numpy
        mask_np = mask[0].cpu().numpy()
        
        # Calculate measurements
        measurements = calculate_segmentation_measurements(mask_np)
        
        return {
            "segmentation_mask": mask_np.tolist(),
            "measurements": measurements
        }

def calculate_confidence(results):
    """Calculate overall confidence score."""
    if "top_prediction" in results:
        return results["top_prediction"]["confidence"]
    elif "measurements" in results:
        # For segmentation, use area confidence
        return min(1.0, results["measurements"]["area"] / 1000.0)
    return 0.0

def filter_by_confidence(results, threshold):
    """Filter results by confidence threshold."""
    if "predictions" in results:
        filtered_predictions = [
            pred for pred in results["predictions"] 
            if pred["confidence"] >= threshold
        ]
        results["predictions"] = filtered_predictions
        
        # Update top prediction if needed
        if results["top_prediction"]["confidence"] < threshold:
            if filtered_predictions:
                results["top_prediction"] = max(filtered_predictions, key=lambda x: x["confidence"])
            else:
                results["top_prediction"] = {"class": "No confident prediction", "confidence": 0.0}
    
    return results

def calculate_segmentation_measurements(mask):
    """Calculate measurements from segmentation mask."""
    # Convert to binary mask
    binary_mask = (mask > 0.5).astype(np.uint8)
    
    # Calculate area
    area = np.sum(binary_mask)
    
    # Calculate bounding box
    coords = np.where(binary_mask)
    if len(coords[0]) > 0:
        bbox = {
            "x_min": int(np.min(coords[1])),
            "y_min": int(np.min(coords[0])),
            "x_max": int(np.max(coords[1])),
            "y_max": int(np.max(coords[0]))
        }
        bbox_area = (bbox["x_max"] - bbox["x_min"]) * (bbox["y_max"] - bbox["y_min"])
    else:
        bbox = {"x_min": 0, "y_min": 0, "x_max": 0, "y_max": 0}
        bbox_area = 0
    
    return {
        "area": int(area),
        "bounding_box": bbox,
        "bounding_box_area": int(bbox_area),
        "density": float(area / max(bbox_area, 1))
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "timestamp": datetime.utcnow().isoformat()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "timestamp": datetime.utcnow().isoformat()}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
