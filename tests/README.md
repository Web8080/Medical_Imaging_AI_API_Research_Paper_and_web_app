# 🔧 Demo and Testing Scripts

This folder contains scripts for demonstrating the Medical Imaging AI system and testing its functionality.

## 📋 Available Scripts

### `simple_pipeline_demo.py`
**Purpose**: Complete pipeline demonstration with mock predictions
**Use Case**: Show how the system works end-to-end
**Features**:
- Image preprocessing
- Mock AI predictions
- Results visualization
- Performance metrics

**Usage**:
```bash
python simple_pipeline_demo.py
```

### `test_complete_pipeline.py`
**Purpose**: End-to-end testing with real trained models
**Use Case**: Test the complete system with actual models
**Features**:
- Real model loading
- Actual predictions
- Comprehensive testing
- Error handling

**Usage**:
```bash
python test_complete_pipeline.py
```

### `test_api.py`
**Purpose**: API endpoint testing
**Use Case**: Test API functionality and endpoints
**Features**:
- Health check testing
- Endpoint validation
- Response verification

**Usage**:
```bash
python test_api.py
```

## 🎯 When to Use Each Script

- **`simple_pipeline_demo.py`**: For demonstrations and showing system capabilities
- **`test_complete_pipeline.py`**: For comprehensive system testing
- **`test_api.py`**: For API-specific testing and validation

## 📊 Expected Outputs

All scripts generate:
- Console output with progress and results
- Visualization files (PNG charts)
- Performance metrics and statistics
- Error reports if issues occur
