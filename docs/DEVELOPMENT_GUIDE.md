# Development Guide

## Getting Started

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Git

### Quick Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd API_for_Medical_Imaging
   ```

2. **Run the setup script:**
   ```bash
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   ```

3. **Start the development server:**
   ```bash
   source venv/bin/activate
   make run-dev
   ```

4. **Access the API:**
   - API: http://localhost:8000
   - Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/api/v1/health

## Project Structure

```
API_for_Medical_Imaging/
├── src/                          # Source code
│   ├── api/                      # API endpoints
│   │   └── v1/                   # API version 1
│   │       ├── endpoints/        # Individual endpoint modules
│   │       └── api.py           # API router configuration
│   ├── core/                     # Core functionality
│   │   ├── config.py            # Configuration management
│   │   ├── database.py          # Database configuration
│   │   └── security.py          # Security utilities
│   ├── models/                   # Database models
│   │   └── database.py          # SQLAlchemy models
│   ├── schemas/                  # Pydantic schemas
│   │   └── api.py               # API request/response models
│   ├── services/                 # Business logic
│   │   ├── dicom_processor.py   # DICOM processing
│   │   ├── model_service.py     # Model inference
│   │   └── job_service.py       # Job management
│   └── main.py                  # Application entry point
├── tests/                        # Test suite
│   ├── conftest.py              # Test configuration
│   ├── test_api.py              # API tests
│   └── test_services.py         # Service tests
├── docs/                         # Documentation
├── scripts/                      # Utility scripts
├── docker-compose.yml           # Docker services
├── Dockerfile                   # Docker image
├── requirements.txt             # Python dependencies
└── Makefile                     # Development commands
```

## Development Workflow

### 1. Environment Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Database Setup

For local development, use Docker Compose:

```bash
# Start database services
docker-compose up -d postgres redis

# Run migrations
make db-migrate
```

### 3. Running Tests

```bash
# Run all tests
make test

# Run tests with coverage
make test-cov

# Run specific test file
pytest tests/test_api.py -v
```

### 4. Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Run type checking
mypy src/
```

### 5. Development Server

```bash
# Run with auto-reload
make run-dev

# Run production server
make run
```

## Adding New Features

### 1. Database Models

Add new models in `src/models/database.py`:

```python
class NewModel(Base):
    __tablename__ = "new_table"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
```

### 2. API Schemas

Define request/response schemas in `src/schemas/api.py`:

```python
class NewModelRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)

class NewModelResponse(BaseModel):
    id: int
    name: str
    created_at: datetime
```

### 3. Service Layer

Implement business logic in `src/services/`:

```python
class NewService:
    def create_item(self, data: NewModelRequest) -> NewModelResponse:
        # Implementation
        pass
```

### 4. API Endpoints

Create endpoints in `src/api/v1/endpoints/`:

```python
@router.post("/items", response_model=NewModelResponse)
async def create_item(
    item: NewModelRequest,
    db: Session = Depends(get_db)
) -> NewModelResponse:
    service = NewService()
    return service.create_item(item)
```

### 5. Tests

Add tests in `tests/`:

```python
def test_create_item(client: TestClient):
    response = client.post("/api/v1/items", json={"name": "test"})
    assert response.status_code == 200
    assert response.json()["name"] == "test"
```

## Configuration

### Environment Variables

Copy `env.example` to `.env` and configure:

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/db

# Security
SECRET_KEY=your-secret-key

# API
API_V1_STR=/api/v1
PROJECT_NAME=Medical Imaging AI API
```

### Model Configuration

Models are configured in the database via the `ModelMetadata` table:

```python
model_config = {
    "type": "pytorch",
    "task_type": "segmentation",
    "input_size": [256, 256, 128],
    "threshold": 0.5,
    "preprocessing": ["normalize", "resize"],
    "postprocessing": ["morphology", "filter"]
}
```

## Data Processing Pipeline

### 1. File Upload

Files are validated and stored temporarily:

```python
# File validation
if not validate_file_upload(file_content, filename):
    raise ValueError("Invalid file")

# Format detection
file_format = detect_format(filename, file_content)
```

### 2. DICOM Processing

DICOM files are processed to extract metadata and pixel data:

```python
processor = DICOMProcessor()
processed_data = processor.process_upload(file_content, filename)
```

### 3. Model Inference

Preprocessed data is sent to AI models:

```python
model_service = ModelService()
results = model_service.predict(model_id, preprocessed_data, config)
```

### 4. Post-processing

Results are processed into standardized format:

```python
detections = postprocess_predictions(results, original_image, config)
```

## Testing

### Test Structure

- `conftest.py`: Test configuration and fixtures
- `test_api.py`: API endpoint tests
- `test_services.py`: Service layer tests

### Running Tests

```bash
# All tests
pytest

# Specific test
pytest tests/test_api.py::test_upload_endpoint

# With coverage
pytest --cov=src --cov-report=html
```

### Test Data

Use fixtures for consistent test data:

```python
@pytest.fixture
def sample_dicom_file():
    # Create test DICOM file
    pass
```

## Docker Development

### Local Development

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Access container
docker-compose exec api bash
```

### Building Images

```bash
# Build API image
docker build -t medical-imaging-api .

# Build with specific tag
docker build -t medical-imaging-api:v1.0.0 .
```

## Deployment

### Production Environment

1. **Environment Configuration:**
   ```bash
   # Set production environment variables
   export DEBUG=False
   export DATABASE_URL=postgresql://prod_user:pass@prod_host:5432/prod_db
   ```

2. **Database Migration:**
   ```bash
   alembic upgrade head
   ```

3. **Start Services:**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

### Health Checks

Monitor API health:

```bash
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/health/detailed
```

## Performance Optimization

### Database Optimization

- Use database indexes for frequently queried fields
- Implement connection pooling
- Use read replicas for read-heavy workloads

### Caching

- Redis for session and job status caching
- Model output caching for identical requests
- CDN for static content

### Monitoring

- Application metrics with Prometheus
- Log aggregation with ELK stack
- Error tracking with Sentry

## Security Considerations

### Data Protection

- Encrypt sensitive data at rest and in transit
- Implement proper access controls
- Regular security audits

### API Security

- Rate limiting
- Input validation
- SQL injection prevention
- XSS protection

### Compliance

- HIPAA compliance for medical data
- GDPR compliance for EU users
- Audit logging for all operations

## Troubleshooting

### Common Issues

1. **Database Connection Errors:**
   ```bash
   # Check database status
   docker-compose ps postgres
   
   # View database logs
   docker-compose logs postgres
   ```

2. **Model Loading Errors:**
   ```bash
   # Check model files
   ls -la models/
   
   # Verify model configuration
   python -c "from src.services.model_service import ModelService; print(ModelService().get_available_models())"
   ```

3. **File Upload Issues:**
   ```bash
   # Check file size limits
   # Verify file format support
   # Check disk space
   df -h
   ```

### Debug Mode

Enable debug mode for detailed error information:

```bash
export DEBUG=True
export LOG_LEVEL=DEBUG
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run the test suite
5. Submit a pull request

### Code Style

- Follow PEP 8
- Use type hints
- Write docstrings
- Add tests for new features

### Commit Messages

Use conventional commit format:

```
feat: add new tumor detection model
fix: resolve DICOM parsing error
docs: update API documentation
test: add integration tests
```

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Pydantic Documentation](https://pydantic-docs.helpmanual.io/)
- [Docker Documentation](https://docs.docker.com/)
- [Medical Imaging Standards](https://www.dicomstandard.org/)
