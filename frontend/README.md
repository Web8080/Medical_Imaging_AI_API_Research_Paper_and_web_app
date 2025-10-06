# Medical Imaging AI Frontend

This directory contains the frontend interfaces for the Medical Imaging AI API, including both Streamlit and React-based dashboards.

## Components

### 1. Streamlit Dashboard (`streamlit_dashboard.py`)
A comprehensive Streamlit-based dashboard for quick prototyping and demonstrations.

**Features:**
- Image upload and analysis
- Real-time results visualization
- System metrics monitoring
- Results history tracking
- Interactive configuration

**Usage:**
```bash
cd frontend
pip install -r requirements.txt
streamlit run streamlit_dashboard.py
```

### 2. React Web Application (`react-app/`)
A full-featured React-based web application with advanced UI components.

**Features:**
- Modern, responsive design
- DICOM viewer with Cornerstone.js
- Advanced visualization with Plotly
- Real-time system monitoring
- Export functionality
- Settings management

**Usage:**
```bash
cd frontend/react-app
npm install
npm start
```

## Quick Start

### Streamlit Dashboard
1. Install dependencies: `pip install -r requirements.txt`
2. Start the dashboard: `streamlit run streamlit_dashboard.py`
3. Open browser to `http://localhost:8501`

### React Application
1. Install Node.js and npm
2. Install dependencies: `npm install`
3. Start development server: `npm start`
4. Open browser to `http://localhost:3000`

## Configuration

Both dashboards connect to the Medical Imaging AI API running on `localhost:8000` by default. You can configure the API endpoint in the settings.

## Features

### Image Analysis
- Support for multiple image formats (PNG, JPG, DICOM, NIfTI)
- Real-time AI analysis with confidence scores
- Multiple model types (Chest, Dermatology, OCT)
- Classification and segmentation capabilities

### DICOM Viewer
- Full DICOM file support
- Window/Level adjustment
- Zoom and pan controls
- Rotation capabilities
- Measurement tools

### System Monitoring
- Real-time API health monitoring
- Performance metrics visualization
- Request history tracking
- Error monitoring and reporting

### Data Export
- JSON export of analysis results
- Image download capabilities
- CSV export of metrics data
- PDF report generation (planned)

## API Integration

Both frontends integrate with the Medical Imaging AI API endpoints:

- `POST /analyze` - Image analysis
- `GET /models` - Available models
- `GET /metrics` - System metrics
- `GET /health` - Health check
- `POST /feedback` - User feedback

## Development

### Streamlit Development
```bash
# Install development dependencies
pip install -r requirements.txt

# Run with auto-reload
streamlit run streamlit_dashboard.py --server.runOnSave true
```

### React Development
```bash
# Install dependencies
npm install

# Start development server
npm start

# Build for production
npm run build
```

## Deployment

### Streamlit Deployment
The Streamlit dashboard can be deployed to:
- Streamlit Cloud
- Docker containers
- Cloud platforms (AWS, GCP, Azure)

### React Deployment
The React app can be deployed to:
- Netlify
- Vercel
- AWS S3 + CloudFront
- Docker containers

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## License

MIT License - see LICENSE file for details.
