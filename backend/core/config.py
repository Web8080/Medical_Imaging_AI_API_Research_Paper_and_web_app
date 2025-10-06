"""
Configuration management for the Medical Imaging AI API.
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, validator


class Settings(BaseSettings):
    """Application settings."""
    
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Medical Imaging AI API"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    RELOAD: bool = False
    
    # Security
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Database
    DATABASE_URL: str
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # AWS Configuration
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: Optional[str] = None
    
    # Model Configuration
    MODEL_CACHE_SIZE: int = 100
    MAX_FILE_SIZE_MB: int = 500
    SUPPORTED_FORMATS: str = "DICOM,NIfTI,JPEG,PNG"
    
    # Monitoring
    SENTRY_DSN: Optional[str] = None
    LOG_LEVEL: str = "INFO"
    
    @validator("SUPPORTED_FORMATS")
    def parse_supported_formats(cls, v: str) -> List[str]:
        """Parse supported formats string into list."""
        return [fmt.strip().upper() for fmt in v.split(",")]
    
    @validator("MAX_FILE_SIZE_MB")
    def validate_file_size(cls, v: int) -> int:
        """Validate maximum file size."""
        if v <= 0:
            raise ValueError("MAX_FILE_SIZE_MB must be positive")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
