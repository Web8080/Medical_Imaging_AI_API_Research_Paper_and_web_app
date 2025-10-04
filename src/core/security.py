"""
Security utilities for authentication and data protection.
"""

import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union

from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from .config import settings


class Token(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str


class TokenData(BaseModel):
    """Token data model."""
    username: Optional[str] = None


# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def create_access_token(
    subject: Union[str, Any], expires_delta: Optional[timedelta] = None
) -> str:
    """Create access token."""
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    
    to_encode = {"exp": expire, "sub": str(subject)}
    encoded_jwt = jwt.encode(
        to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM
    )
    return encoded_jwt


def verify_token(token: str) -> Optional[str]:
    """Verify and decode token."""
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        username: str = payload.get("sub")
        if username is None:
            return None
        return username
    except JWTError:
        return None


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)


def generate_api_key() -> str:
    """Generate a secure API key."""
    return secrets.token_urlsafe(32)


def anonymize_dicom_data(dicom_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Anonymize DICOM data by removing patient identifying information.
    
    This function removes or replaces sensitive patient information
    to ensure HIPAA compliance.
    """
    # List of DICOM tags that contain patient identifying information
    patient_tags = [
        "PatientName",
        "PatientID", 
        "PatientBirthDate",
        "PatientSex",
        "PatientAge",
        "PatientAddress",
        "PatientTelephoneNumbers",
        "PatientMotherBirthName",
        "PatientWeight",
        "PatientSize",
        "PatientComments",
        "StudyDate",
        "StudyTime",
        "AccessionNumber",
        "StudyID",
        "StudyDescription",
        "SeriesDescription",
        "InstitutionName",
        "InstitutionAddress",
        "ReferringPhysicianName",
        "PerformingPhysicianName",
        "OperatorName",
        "Manufacturer",
        "ManufacturerModelName",
        "DeviceSerialNumber",
        "SoftwareVersions"
    ]
    
    anonymized_data = dicom_data.copy()
    
    for tag in patient_tags:
        if tag in anonymized_data:
            if tag in ["PatientID", "StudyID", "AccessionNumber"]:
                # Replace with anonymized ID
                anonymized_data[tag] = f"ANON_{secrets.token_hex(8)}"
            elif tag in ["StudyDate", "PatientBirthDate"]:
                # Replace with anonymized date (keep year for age calculation)
                original_date = anonymized_data[tag]
                if original_date and len(str(original_date)) >= 4:
                    anonymized_data[tag] = f"1900{str(original_date)[4:]}"
            else:
                # Remove or replace with generic value
                anonymized_data[tag] = "ANONYMIZED"
    
    return anonymized_data


def validate_file_upload(file_content: bytes, filename: str) -> bool:
    """
    Validate uploaded file for security and format compliance.
    
    Args:
        file_content: The file content as bytes
        filename: The original filename
        
    Returns:
        True if file is valid, False otherwise
    """
    # Check file size
    max_size = settings.MAX_FILE_SIZE_MB * 1024 * 1024  # Convert to bytes
    if len(file_content) > max_size:
        return False
    
    # Check file extension
    allowed_extensions = ['.dcm', '.dicom', '.nii', '.nii.gz', '.jpg', '.jpeg', '.png']
    file_ext = filename.lower().split('.')[-1]
    if f'.{file_ext}' not in allowed_extensions:
        return False
    
    # Basic file content validation
    if len(file_content) < 100:  # Minimum file size
        return False
    
    return True
