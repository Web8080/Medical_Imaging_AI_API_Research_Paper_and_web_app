#!/usr/bin/env python3
"""
Medical Imaging AI - Quick Start Launcher
Simple script to start the system and open the UI
"""

import subprocess
import time
import webbrowser
import sys
import os

def main():
    """Quick start the Medical Imaging AI system"""
    print("=" * 60)
    print("🏥 MEDICAL IMAGING AI - QUICK START")
    print("=" * 60)
    
    print("🚀 Starting services...")
    
    # Start API server
    print("📡 Starting API server...")
    api_process = subprocess.Popen([
        sys.executable, "organized_scripts/api_servers/simple_api_server.py"
    ], cwd=os.getcwd())
    
    time.sleep(3)
    
    # Start Streamlit dashboard
    print("📊 Starting Streamlit dashboard...")
    streamlit_process = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", 
        "frontend/streamlit_dashboard.py", 
        "--server.port", "8501", 
        "--server.address", "0.0.0.0"
    ], cwd=os.getcwd())
    
    time.sleep(5)
    
    # Open browser
    print("🌐 Opening browser...")
    webbrowser.open("http://localhost:8501")
    
    print("\n" + "=" * 60)
    print("✅ SYSTEM READY!")
    print("=" * 60)
    print("🌐 Streamlit Dashboard: http://localhost:8501")
    print("📡 API Server: http://localhost:8001")
    print("🔍 Health Check: http://localhost:8001/health")
    
    print("\n🎯 FEATURES AVAILABLE:")
    print("• Upload medical images (PNG, JPG, DICOM)")
    print("• AI-powered analysis and predictions")
    print("• Interactive results visualization")
    print("• DICOM image viewer")
    print("• Results history")
    
    print("\n📁 ORGANIZED SCRIPTS:")
    print("• Demo & Testing: organized_scripts/demo_and_testing/")
    print("• UI Launchers: organized_scripts/ui_launchers/")
    print("• API Servers: organized_scripts/api_servers/")
    print("• Data Processing: organized_scripts/data_processing/")
    print("• Model Training: organized_scripts/model_training/")
    print("• Visualizations: organized_scripts/visualization/")
    
    print("\n🎉 Ready to analyze medical images!")
    
    try:
        print("\nPress Ctrl+C to stop all services...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        api_process.terminate()
        streamlit_process.terminate()
        print("✅ Services stopped.")

if __name__ == "__main__":
    main()
