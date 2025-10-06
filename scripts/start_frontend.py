#!/usr/bin/env python3
"""
Frontend Startup Script for Medical Imaging AI Dashboard
Launches both Streamlit and React frontends
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path
import threading
import os

def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    
    # Check Python packages
    try:
        import streamlit
        print("✓ Streamlit installed")
    except ImportError:
        print("✗ Streamlit not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit", "plotly", "pandas", "pillow", "requests"])
    
    # Check Node.js
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Node.js installed: {result.stdout.strip()}")
        else:
            print("✗ Node.js not found. Please install Node.js from https://nodejs.org/")
            return False
    except FileNotFoundError:
        print("✗ Node.js not found. Please install Node.js from https://nodejs.org/")
        return False
    
    # Check npm
    try:
        result = subprocess.run(["npm", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ npm installed: {result.stdout.strip()}")
        else:
            print("✗ npm not found")
            return False
    except FileNotFoundError:
        print("✗ npm not found")
        return False
    
    return True

def install_react_dependencies():
    """Install React app dependencies."""
    print("Installing React dependencies...")
    react_dir = Path("frontend/react-app")
    
    if not react_dir.exists():
        print("✗ React app directory not found")
        return False
    
    try:
        subprocess.run(["npm", "install"], cwd=react_dir, check=True)
        print("✓ React dependencies installed")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to install React dependencies")
        return False

def start_streamlit():
    """Start Streamlit dashboard."""
    print("Starting Streamlit dashboard...")
    streamlit_file = Path("frontend/streamlit_dashboard.py")
    
    if not streamlit_file.exists():
        print("✗ Streamlit dashboard file not found")
        return
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(streamlit_file),
            "--server.port", "8501",
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("Streamlit dashboard stopped")

def start_react():
    """Start React development server."""
    print("Starting React development server...")
    react_dir = Path("frontend/react-app")
    
    if not react_dir.exists():
        print("✗ React app directory not found")
        return
    
    try:
        subprocess.run(["npm", "start"], cwd=react_dir)
    except KeyboardInterrupt:
        print("React development server stopped")

def open_browsers():
    """Open browsers to the frontend applications."""
    time.sleep(3)  # Wait for servers to start
    
    print("Opening browsers...")
    try:
        webbrowser.open("http://localhost:8501")  # Streamlit
        time.sleep(1)
        webbrowser.open("http://localhost:3000")  # React
    except Exception as e:
        print(f"Could not open browsers: {e}")

def main():
    """Main function to start both frontends."""
    print("=" * 60)
    print("Medical Imaging AI Frontend Startup")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("Please install missing dependencies and try again.")
        return
    
    # Install React dependencies
    if not install_react_dependencies():
        print("Please fix React dependencies and try again.")
        return
    
    print("\nStarting frontend applications...")
    print("Streamlit Dashboard: http://localhost:8501")
    print("React Application: http://localhost:3000")
    print("\nPress Ctrl+C to stop all servers")
    print("=" * 60)
    
    # Start browsers in a separate thread
    browser_thread = threading.Thread(target=open_browsers)
    browser_thread.daemon = True
    browser_thread.start()
    
    try:
        # Start both servers
        streamlit_thread = threading.Thread(target=start_streamlit)
        react_thread = threading.Thread(target=start_react)
        
        streamlit_thread.daemon = True
        react_thread.daemon = True
        
        streamlit_thread.start()
        react_thread.start()
        
        # Wait for threads
        streamlit_thread.join()
        react_thread.join()
        
    except KeyboardInterrupt:
        print("\nShutting down frontend servers...")
        print("Thank you for using Medical Imaging AI Dashboard!")

if __name__ == "__main__":
    main()
