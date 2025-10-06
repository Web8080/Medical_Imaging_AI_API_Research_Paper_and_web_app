#!/usr/bin/env python3
"""
Medical Imaging AI API - Streamlit Dashboard
Phase 6: Frontend Dashboard for testing and demonstration

A comprehensive Streamlit dashboard for interacting with the Medical Imaging AI API.
"""

import streamlit as st
import requests
import json
import time
import io
import base64
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, Any, Optional
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Medical Imaging AI Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8001"
API_TOKEN = "test_token"  # This should be configurable

class MedicalAIDashboard:
    """Main dashboard class for Medical Imaging AI API."""
    
    def __init__(self):
        self.api_base_url = API_BASE_URL
        self.api_token = API_TOKEN
        
    def check_api_health(self) -> bool:
        """Check if the API is running and healthy."""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get list of available models from the API."""
        try:
            headers = {"Authorization": f"Bearer {self.api_token}"}
            response = requests.get(f"{self.api_base_url}/models", headers=headers)
            if response.status_code == 200:
                return response.json()
            return {"available_models": [], "model_info": {}}
        except:
            return {"available_models": [], "model_info": {}}
    
    def analyze_image(self, image_file, model_type: str, analysis_type: str, confidence_threshold: float) -> Dict[str, Any]:
        """Send image for analysis to the API."""
        try:
            headers = {"Authorization": f"Bearer {self.api_token}"}
            files = {"file": image_file}
            data = {
                "model_type": model_type,
                "analysis_type": analysis_type,
                "confidence_threshold": confidence_threshold
            }
            
            response = requests.post(
                f"{self.api_base_url}/upload",
                files=files,
                data=data,
                headers=headers
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API Error: {response.status_code} - {response.text}"}
        except Exception as e:
            return {"error": f"Connection Error: {str(e)}"}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics from the API."""
        try:
            headers = {"Authorization": f"Bearer {self.api_token}"}
            response = requests.get(f"{self.api_base_url}/metrics", headers=headers)
            if response.status_code == 200:
                return response.json()
            return {}
        except:
            return {}

def main():
    """Main dashboard application."""
    
    # Initialize dashboard
    dashboard = MedicalAIDashboard()
    
    # Title and description
    st.title("Medical Imaging AI Dashboard")
    st.markdown("""
    A comprehensive dashboard for interacting with the Medical Imaging AI API.
    Upload medical images to get AI-powered analysis including tumor detection and classification.
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Status
        api_healthy = dashboard.check_api_health()
        if api_healthy:
            st.success("API Status: Online")
        else:
            st.error("API Status: Offline")
            st.warning("Please ensure the API server is running on localhost:8001")
            return
        
        # Model selection
        st.subheader("Model Configuration")
        models_data = dashboard.get_available_models()
        available_models = models_data.get("available_models", ["chest", "derma", "oct"])
        
        model_type = st.selectbox(
            "Select Model Type",
            available_models,
            help="Choose the AI model for analysis"
        )
        
        analysis_type = st.selectbox(
            "Analysis Type",
            ["classification", "segmentation"],
            help="Choose between classification or segmentation analysis"
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Minimum confidence level for predictions"
        )
        
        # API Token configuration
        st.subheader("API Configuration")
        api_token = st.text_input(
            "API Token",
            value=API_TOKEN,
            type="password",
            help="Enter your API authentication token"
        )
        dashboard.api_token = api_token
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["Image Analysis", "Results History", "System Metrics", "Documentation"])
    
    with tab1:
        st.header("Medical Image Analysis")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Medical Image",
            type=['png', 'jpg', 'jpeg', 'dcm', 'nii', 'nii.gz'],
            help="Upload a medical image for AI analysis"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Image information
                st.info(f"""
                **Image Information:**
                - File: {uploaded_file.name}
                - Size: {image.size}
                - Mode: {image.mode}
                - Format: {uploaded_file.type}
                """)
            
            with col2:
                st.subheader("Analysis Configuration")
                st.write(f"**Model Type:** {model_type}")
                st.write(f"**Analysis Type:** {analysis_type}")
                st.write(f"**Confidence Threshold:** {confidence_threshold}")
                
                # Analyze button
                if st.button("Analyze Image", type="primary"):
                    with st.spinner("Analyzing image..."):
                        # Reset file pointer
                        uploaded_file.seek(0)
                        
                        # Send for analysis
                        result = dashboard.analyze_image(
                            uploaded_file,
                            model_type,
                            analysis_type,
                            confidence_threshold
                        )
                        
                        if "error" in result:
                            st.error(f"Analysis failed: {result['error']}")
                        else:
                            st.success("Analysis completed successfully!")
                            
                            # Store result in session state
                            if "analysis_results" not in st.session_state:
                                st.session_state.analysis_results = []
                            st.session_state.analysis_results.append(result)
                            
                            # Display results
                            display_analysis_results(result)
    
    with tab2:
        st.header("Analysis Results History")
        
        if "analysis_results" in st.session_state and st.session_state.analysis_results:
            for i, result in enumerate(reversed(st.session_state.analysis_results)):
                with st.expander(f"Analysis {len(st.session_state.analysis_results) - i} - {result.get('timestamp', 'Unknown time')}"):
                    display_analysis_results(result)
        else:
            st.info("No analysis results yet. Upload and analyze an image to see results here.")
    
    with tab3:
        st.header("System Metrics")
        
        if st.button("Refresh Metrics"):
            metrics = dashboard.get_metrics()
            
            if metrics:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Requests", metrics.get("total_requests", 0))
                
                with col2:
                    st.metric("Successful Requests", metrics.get("successful_requests", 0))
                
                with col3:
                    st.metric("Failed Requests", metrics.get("failed_requests", 0))
                
                with col4:
                    avg_time = metrics.get("average_processing_time", 0)
                    st.metric("Avg Processing Time", f"{avg_time:.2f}s")
                
                # Model performance chart
                if "model_performance" in metrics:
                    st.subheader("Model Performance")
                    model_perf = metrics["model_performance"]
                    
                    if model_perf:
                        df = pd.DataFrame([
                            {
                                "Model": model,
                                "Success Rate": data.get("success_rate", 0) * 100,
                                "Avg Processing Time": data.get("average_processing_time", 0)
                            }
                            for model, data in model_perf.items()
                        ])
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig = px.bar(df, x="Model", y="Success Rate", title="Model Success Rates")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            fig = px.bar(df, x="Model", y="Avg Processing Time", title="Average Processing Times")
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Unable to retrieve metrics. Check API connection.")
        else:
            st.info("Click 'Refresh Metrics' to view system performance data.")
    
    with tab4:
        st.header("Documentation")
        
        st.markdown("""
        ## Medical Imaging AI API Dashboard
        
        This dashboard provides a user-friendly interface for interacting with the Medical Imaging AI API.
        
        ### Features
        
        - **Image Upload**: Support for various medical image formats (PNG, JPG, DICOM, NIfTI)
        - **Model Selection**: Choose from different AI models (Chest, Dermatology, OCT)
        - **Analysis Types**: Classification and segmentation capabilities
        - **Real-time Results**: Instant analysis results with confidence scores
        - **System Monitoring**: View API performance metrics and statistics
        
        ### Supported Models
        
        1. **Chest Model**: Analyzes chest X-ray images for pathology detection
        2. **Dermatology Model**: Classifies skin lesions from dermatoscopic images
        3. **OCT Model**: Analyzes retinal OCT images for disease detection
        
        ### API Endpoints
        
        - `POST /analyze` - Upload and analyze medical images
        - `GET /models` - List available models
        - `GET /metrics` - Get system performance metrics
        - `GET /health` - Check API health status
        
        ### Getting Started
        
        1. Ensure the API server is running on localhost:8000
        2. Upload a medical image using the file uploader
        3. Configure analysis parameters in the sidebar
        4. Click "Analyze Image" to get results
        5. View results in the main panel and history tab
        
        ### Troubleshooting
        
        - **API Offline**: Make sure the API server is running
        - **Authentication Error**: Check your API token in the sidebar
        - **Analysis Failed**: Verify image format and model compatibility
        """)

def display_analysis_results(result: Dict[str, Any]):
    """Display analysis results in a formatted way."""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Analysis Results")
        
        # Basic information
        st.write(f"**Request ID:** {result.get('request_id', 'N/A')}")
        st.write(f"**Model Used:** {result.get('model_used', 'N/A')}")
        processing_time = result.get('processing_time', '0s')
        if isinstance(processing_time, str):
            st.write(f"**Processing Time:** {processing_time}")
        else:
            st.write(f"**Processing Time:** {processing_time:.2f} seconds")
        confidence = result.get('prediction', {}).get('confidence', 0)
        st.write(f"**Confidence:** {confidence:.3f}")
        st.write(f"**Timestamp:** {result.get('timestamp', 'N/A')}")
        
        # Results
        prediction = result.get('prediction', {})
        
        if 'predicted_class' in prediction:
            st.subheader("Top Prediction")
            predicted_class = prediction.get('predicted_class', 'Unknown')
            confidence = prediction.get('confidence', 0)
            st.success(f"**{predicted_class}** (Confidence: {confidence:.3f})")
        
        if 'top5_predictions' in prediction:
            st.subheader("Top 5 Predictions")
            predictions = prediction['top5_predictions']
            
            if predictions:
                df = pd.DataFrame(predictions)
                df = df.sort_values('probability', ascending=False)
                df['probability'] = df['probability'].round(3)
                df.columns = ['Class', 'Probability']
                
                st.dataframe(df, use_container_width=True)
                
                # Confidence chart
                fig = px.bar(df, x='Class', y='Probability', title='Prediction Confidence Scores')
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        
        if 'measurements' in results:
            st.subheader("Segmentation Measurements")
            measurements = results['measurements']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Area", f"{measurements.get('area', 0):,} pixels")
            with col2:
                st.metric("Density", f"{measurements.get('density', 0):.3f}")
            with col3:
                bbox = measurements.get('bounding_box', {})
                st.metric("Bounding Box Area", f"{measurements.get('bounding_box_area', 0):,} pixels")
    
    with col2:
        st.subheader("Result Summary")
        
        # Status indicator
        status = result.get('status', 'unknown')
        if status == 'success':
            st.success("Analysis Successful")
        else:
            st.error(f"Status: {status}")
        
        # Confidence indicator
        confidence = result.get('confidence', 0)
        if confidence > 0.8:
            st.success(f"High Confidence: {confidence:.3f}")
        elif confidence > 0.5:
            st.warning(f"Medium Confidence: {confidence:.3f}")
        else:
            st.error(f"Low Confidence: {confidence:.3f}")
        
        # Model information
        model_info = prediction.get('model_info', {})
        if model_info:
            st.info(f"**Model:** {model_info.get('name', 'Unknown')} - {model_info.get('description', 'No description')}")
        else:
            st.info("**Model:** Mock Advanced CNN - Demonstration model for medical image classification")

if __name__ == "__main__":
    main()
