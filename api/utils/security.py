#!/usr/bin/env python3
"""
Security Manager for Medical Imaging AI API
Handles authentication, authorization, and security measures.
"""

import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Set
import jwt
import os

logger = logging.getLogger(__name__)

class SecurityManager:
    """Handles security operations for the API."""
    
    def __init__(self):
        self.secret_key = os.getenv('API_SECRET_KEY', secrets.token_hex(32))
        self.token_expiry = timedelta(hours=24)
        self.rate_limit_requests = 100  # requests per hour
        self.rate_limit_window = 3600  # 1 hour in seconds
        
        # In production, these would be stored in a secure database
        self.valid_tokens: Set[str] = set()
        self.user_requests: Dict[str, list] = {}
        
        # Generate default API key for testing
        self._generate_default_token()
    
    def _generate_default_token(self):
        """Generate a default API token for testing."""
        default_token = self.generate_token("test_user")
        self.valid_tokens.add(default_token)
        logger.info(f"Generated default API token: {default_token}")
    
    def generate_token(self, user_id: str) -> str:
        """Generate a new API token."""
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + self.token_expiry,
            'iat': datetime.utcnow()
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm='HS256')
        self.valid_tokens.add(token)
        
        logger.info(f"Generated new token for user: {user_id}")
        return token
    
    def verify_token(self, token: str) -> bool:
        """Verify if a token is valid."""
        try:
            # Check if token is in valid tokens set
            if token not in self.valid_tokens:
                return False
            
            # Decode and verify JWT
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            
            # Check expiration
            if datetime.utcnow() > datetime.fromtimestamp(payload['exp']):
                self.valid_tokens.discard(token)
                return False
            
            return True
            
        except jwt.ExpiredSignatureError:
            self.valid_tokens.discard(token)
            return False
        except jwt.InvalidTokenError:
            return False
        except Exception as e:
            logger.error(f"Error verifying token: {str(e)}")
            return False
    
    def revoke_token(self, token: str) -> bool:
        """Revoke a token."""
        if token in self.valid_tokens:
            self.valid_tokens.discard(token)
            logger.info("Token revoked successfully")
            return True
        return False
    
    def check_rate_limit(self, client_ip: str) -> bool:
        """Check if client has exceeded rate limit."""
        current_time = datetime.utcnow()
        
        # Clean old requests
        if client_ip in self.user_requests:
            self.user_requests[client_ip] = [
                req_time for req_time in self.user_requests[client_ip]
                if (current_time - req_time).total_seconds() < self.rate_limit_window
            ]
        else:
            self.user_requests[client_ip] = []
        
        # Check if under limit
        if len(self.user_requests[client_ip]) >= self.rate_limit_requests:
            return False
        
        # Add current request
        self.user_requests[client_ip].append(current_time)
        return True
    
    def hash_file_content(self, content: bytes) -> str:
        """Generate hash for file content."""
        return hashlib.sha256(content).hexdigest()
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for security."""
        # Remove path traversal attempts
        filename = filename.replace('..', '').replace('/', '').replace('\\', '')
        
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*']
        for char in dangerous_chars:
            filename = filename.replace(char, '')
        
        return filename
    
    def validate_file_type(self, filename: str, allowed_extensions: set) -> bool:
        """Validate file type based on extension."""
        if not filename:
            return False
        
        extension = filename.lower().split('.')[-1]
        return extension in allowed_extensions
    
    def get_user_from_token(self, token: str) -> Optional[str]:
        """Extract user ID from token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload.get('user_id')
        except:
            return None
