# 🚀 UI Launcher Scripts

This folder contains scripts to launch the user interface components of the Medical Imaging AI system.

## 📋 Available Scripts

### `launch_ui.py`
**Purpose**: Main launcher for both Streamlit dashboard and API server
**Use Case**: Start the complete system with one command
**Features**:
- Launches Streamlit dashboard
- Starts API server
- Opens browser automatically
- Health checks and status monitoring

**Usage**:
```bash
python launch_ui.py
```

### `start_frontend.py`
**Purpose**: Frontend application starter
**Use Case**: Launch only the frontend components
**Features**:
- Streamlit dashboard
- React web interface
- Dependency checking

**Usage**:
```bash
python start_frontend.py
```

### `start_web_interface.py`
**Purpose**: Web interface launcher with browser opening
**Use Case**: Quick access to the web interface
**Features**:
- Opens browser automatically
- Provides usage instructions
- Shows available features

**Usage**:
```bash
python start_web_interface.py
```

## 🌐 Access URLs

After launching:
- **Streamlit Dashboard**: http://localhost:8501
- **API Server**: http://localhost:8001
- **Health Check**: http://localhost:8001/health

## 🎯 When to Use Each Script

- **`launch_ui.py`**: For complete system startup (recommended)
- **`start_frontend.py`**: For frontend-only testing
- **`start_web_interface.py`**: For quick web access

## 🔧 Troubleshooting

If services fail to start:
1. Check if ports 8501 and 8001 are available
2. Ensure all dependencies are installed
3. Check the console output for error messages
