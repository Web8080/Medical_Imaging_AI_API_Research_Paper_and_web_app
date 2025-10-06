#!/usr/bin/env python3
"""
Medical Imaging AI - UI Launcher
Launches both Streamlit dashboard and API server
"""

import subprocess
import time
import webbrowser
import sys
import os
from pathlib import Path

def check_port(port):
    """Check if a port is available"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0

def launch_streamlit():
    """Launch Streamlit dashboard"""
    print("🚀 Launching Streamlit Dashboard...")
    
    # Check if Streamlit is already running
    if not check_port(8501):
        print("✅ Streamlit is already running on http://localhost:8501")
        return True
    
    try:
        # Launch Streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", "frontend/streamlit_dashboard.py", 
               "--server.port", "8501", "--server.address", "0.0.0.0"]
        
        subprocess.Popen(cmd, cwd=os.getcwd())
        time.sleep(3)
        
        print("✅ Streamlit Dashboard launched successfully!")
        print("🌐 Access at: http://localhost:8501")
        return True
        
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")
        return False

def launch_api():
    """Launch API server"""
    print("🚀 Launching API Server...")
    
    # Check if API is already running
    if not check_port(8001):
        print("✅ API is already running on http://localhost:8001")
        return True
    
    try:
        # Launch API server
        cmd = [sys.executable, "-c", """
import sys
sys.path.append('api')
from main import app
import uvicorn
uvicorn.run(app, host='0.0.0.0', port=8001, log_level='info')
"""]
        
        subprocess.Popen(cmd, cwd=os.getcwd())
        time.sleep(5)
        
        print("✅ API Server launched successfully!")
        print("🌐 Access at: http://localhost:8001")
        return True
        
    except Exception as e:
        print(f"❌ Error launching API: {e}")
        return False

def open_browser():
    """Open browser to the dashboard"""
    try:
        print("🌐 Opening browser...")
        webbrowser.open("http://localhost:8501")
        print("✅ Browser opened to Streamlit dashboard")
    except Exception as e:
        print(f"❌ Error opening browser: {e}")

def main():
    """Main launcher function"""
    print("=" * 60)
    print("🏥 MEDICAL IMAGING AI - UI LAUNCHER")
    print("=" * 60)
    
    # Launch Streamlit
    streamlit_success = launch_streamlit()
    
    # Launch API
    api_success = launch_api()
    
    # Open browser
    if streamlit_success:
        open_browser()
    
    print("\n" + "=" * 60)
    print("📋 ACCESS INFORMATION")
    print("=" * 60)
    
    if streamlit_success:
        print("✅ Streamlit Dashboard: http://localhost:8501")
        print("   - Interactive medical imaging interface")
        print("   - Upload and analyze medical images")
        print("   - View results and visualizations")
    else:
        print("❌ Streamlit Dashboard: Not available")
    
    if api_success:
        print("✅ API Server: http://localhost:8001")
        print("   - RESTful API endpoints")
        print("   - Model inference services")
        print("   - Health check: http://localhost:8001/health")
    else:
        print("❌ API Server: Not available")
    
    print("\n" + "=" * 60)
    print("🎯 FEATURES AVAILABLE")
    print("=" * 60)
    print("📸 Image Upload & Analysis")
    print("🤖 AI Model Predictions")
    print("📊 Results Visualization")
    print("📈 Performance Metrics")
    print("🔍 DICOM Image Viewer")
    print("📋 Results History")
    
    print("\n" + "=" * 60)
    print("🚀 READY TO USE!")
    print("=" * 60)
    print("The Medical Imaging AI system is now running.")
    print("Use the Streamlit dashboard to upload and analyze medical images.")
    
    # Keep the script running
    try:
        print("\nPress Ctrl+C to stop all services...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        print("✅ Services stopped.")

if __name__ == "__main__":
    main()
