#!/usr/bin/env python3
"""
Database Manager for Medical Imaging AI API
Handles data storage, caching, and retrieval.
"""

import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Handles database operations for the API."""
    
    def __init__(self, db_path: str = "api_data.db"):
        self.db_path = db_path
        self.connection = None
        self.cache_ttl = timedelta(hours=24)  # Cache time-to-live
        
    async def initialize(self):
        """Initialize database and create tables."""
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
            
            # Create tables
            await self._create_tables()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
    
    async def _create_tables(self):
        """Create database tables."""
        cursor = self.connection.cursor()
        
        # Results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id TEXT PRIMARY KEY,
                file_hash TEXT,
                model_type TEXT,
                results TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP
            )
        """)
        
        # Requests table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS requests (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                model_type TEXT,
                processing_time REAL,
                success BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT,
                feedback TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (request_id) REFERENCES requests (id)
            )
        """)
        
        # Cache table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP
            )
        """)
        
        self.connection.commit()
    
    async def cache_result(self, file_hash: str, model_type: str, results: Dict[str, Any]):
        """Cache analysis results."""
        try:
            cursor = self.connection.cursor()
            
            # Generate cache key
            cache_key = f"{file_hash}_{model_type}"
            
            # Store in cache table
            cursor.execute("""
                INSERT OR REPLACE INTO cache (key, value, expires_at)
                VALUES (?, ?, ?)
            """, (
                cache_key,
                json.dumps(results),
                datetime.utcnow() + self.cache_ttl
            ))
            
            # Also store in results table
            cursor.execute("""
                INSERT OR REPLACE INTO results (id, file_hash, model_type, results, expires_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                cache_key,
                file_hash,
                model_type,
                json.dumps(results),
                datetime.utcnow() + self.cache_ttl
            ))
            
            self.connection.commit()
            logger.info(f"Cached result for {cache_key}")
            
        except Exception as e:
            logger.error(f"Error caching result: {str(e)}")
    
    async def get_cached_result(self, file_hash: str, model_type: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis result."""
        try:
            cursor = self.connection.cursor()
            cache_key = f"{file_hash}_{model_type}"
            
            cursor.execute("""
                SELECT value FROM cache 
                WHERE key = ? AND expires_at > ?
            """, (cache_key, datetime.utcnow()))
            
            row = cursor.fetchone()
            if row:
                return json.loads(row['value'])
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached result: {str(e)}")
            return None
    
    async def get_result(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get analysis result by request ID."""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                SELECT results FROM results WHERE id = ?
            """, (request_id,))
            
            row = cursor.fetchone()
            if row:
                return json.loads(row['results'])
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting result: {str(e)}")
            return None
    
    async def store_feedback(self, request_id: str, feedback: Dict[str, Any]):
        """Store user feedback."""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                INSERT INTO feedback (request_id, feedback)
                VALUES (?, ?)
            """, (request_id, json.dumps(feedback)))
            
            self.connection.commit()
            logger.info(f"Stored feedback for {request_id}")
            
        except Exception as e:
            logger.error(f"Error storing feedback: {str(e)}")
    
    async def get_feedback(self, request_id: str) -> List[Dict[str, Any]]:
        """Get feedback for a request."""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                SELECT feedback, created_at FROM feedback 
                WHERE request_id = ? ORDER BY created_at DESC
            """, (request_id,))
            
            rows = cursor.fetchall()
            return [json.loads(row['feedback']) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting feedback: {str(e)}")
            return []
    
    async def cleanup_expired_data(self):
        """Clean up expired data."""
        try:
            cursor = self.connection.cursor()
            
            # Clean expired cache entries
            cursor.execute("""
                DELETE FROM cache WHERE expires_at < ?
            """, (datetime.utcnow(),))
            
            # Clean expired results
            cursor.execute("""
                DELETE FROM results WHERE expires_at < ?
            """, (datetime.utcnow(),))
            
            # Clean old requests (keep for 30 days)
            cursor.execute("""
                DELETE FROM requests WHERE created_at < ?
            """, (datetime.utcnow() - timedelta(days=30),))
            
            self.connection.commit()
            logger.info("Cleaned up expired data")
            
        except Exception as e:
            logger.error(f"Error cleaning up data: {str(e)}")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            cursor = self.connection.cursor()
            
            # Count total requests
            cursor.execute("SELECT COUNT(*) as total FROM requests")
            total_requests = cursor.fetchone()['total']
            
            # Count successful requests
            cursor.execute("SELECT COUNT(*) as successful FROM requests WHERE success = 1")
            successful_requests = cursor.fetchone()['successful']
            
            # Count failed requests
            cursor.execute("SELECT COUNT(*) as failed FROM requests WHERE success = 0")
            failed_requests = cursor.fetchone()['failed']
            
            # Average processing time
            cursor.execute("SELECT AVG(processing_time) as avg_time FROM requests WHERE success = 1")
            avg_time = cursor.fetchone()['avg_time'] or 0
            
            return {
                'total_requests': total_requests,
                'successful_requests': successful_requests,
                'failed_requests': failed_requests,
                'average_processing_time': avg_time
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {}
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
