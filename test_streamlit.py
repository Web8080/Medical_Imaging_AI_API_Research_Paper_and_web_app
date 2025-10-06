#!/usr/bin/env python3
"""
Simple test Streamlit app to verify Streamlit is working
"""

import streamlit as st

st.set_page_config(
    page_title="Test Streamlit",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Streamlit Test App")
st.write("If you can see this, Streamlit is working!")

st.subheader("API Test")
try:
    import requests
    response = requests.get("http://localhost:8001/health", timeout=5)
    if response.status_code == 200:
        st.success("✅ API is responding!")
        st.json(response.json())
    else:
        st.error(f"❌ API returned status {response.status_code}")
except Exception as e:
    st.error(f"❌ API connection failed: {e}")

st.subheader("File Upload Test")
uploaded_file = st.file_uploader("Choose a file", type=['png', 'jpg', 'jpeg'])
if uploaded_file is not None:
    st.success(f"✅ File uploaded: {uploaded_file.name}")
    st.write(f"Size: {uploaded_file.size} bytes")
    st.write(f"Type: {uploaded_file.type}")

st.subheader("Basic Widgets Test")
col1, col2 = st.columns(2)

with col1:
    st.write("**Text Input:**")
    name = st.text_input("Enter your name", "Test User")
    st.write(f"Hello, {name}!")

with col2:
    st.write("**Selectbox:**")
    option = st.selectbox("Choose an option", ["Option 1", "Option 2", "Option 3"])
    st.write(f"Selected: {option}")

st.subheader("Charts Test")
import pandas as pd
import numpy as np

# Create sample data
data = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100)
})

st.scatter_chart(data)
st.bar_chart(data.head(10))

st.success("🎉 All tests passed! Streamlit is working correctly.")
