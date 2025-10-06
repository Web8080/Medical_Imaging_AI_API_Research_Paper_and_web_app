#!/usr/bin/env python3
"""
Minimal Streamlit dashboard to test basic functionality
"""

import streamlit as st
import requests

st.set_page_config(
    page_title="Minimal Medical AI Dashboard",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Medical Imaging AI Dashboard")
st.write("A comprehensive dashboard for interacting with the Medical Imaging AI API.")

# Test API connection
st.subheader("API Status")
try:
    response = requests.get("http://localhost:8001/health", timeout=5)
    if response.status_code == 200:
        st.success("✅ API Status: Online")
        st.json(response.json())
    else:
        st.error(f"❌ API Status: Error {response.status_code}")
except Exception as e:
    st.error(f"❌ API Status: Offline - {e}")

# Simple file upload
st.subheader("Image Upload Test")
uploaded_file = st.file_uploader(
    "Upload a medical image",
    type=['png', 'jpg', 'jpeg'],
    help="Upload a medical image for analysis"
)

if uploaded_file is not None:
    st.success(f"✅ File uploaded: {uploaded_file.name}")
    st.write(f"Size: {uploaded_file.size} bytes")
    st.write(f"Type: {uploaded_file.type}")
    
    # Test API upload
    if st.button("Analyze Image"):
        try:
            files = {"file": uploaded_file}
            response = requests.post("http://localhost:8001/upload", files=files)
            
            if response.status_code == 200:
                result = response.json()
                st.success("✅ Analysis completed!")
                
                # Display results
                prediction = result.get('prediction', {})
                if prediction:
                    st.write(f"**Predicted Class:** {prediction.get('predicted_class', 'Unknown')}")
                    st.write(f"**Confidence:** {prediction.get('confidence', 0):.3f}")
                    
                    # Show top 5 predictions
                    top5 = prediction.get('top5_predictions', [])
                    if top5:
                        st.write("**Top 5 Predictions:**")
                        for i, pred in enumerate(top5, 1):
                            st.write(f"{i}. {pred['class']}: {pred['probability']:.3f}")
            else:
                st.error(f"❌ Analysis failed: {response.status_code}")
                st.write(response.text)
        except Exception as e:
            st.error(f"❌ Analysis error: {e}")

st.write("---")
st.write("This is a minimal dashboard to test basic functionality.")
