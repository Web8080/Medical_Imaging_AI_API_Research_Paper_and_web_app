#!/usr/bin/env python3
"""
Metrics Calculator for Medical Imaging AI API
Handles performance metrics and monitoring.
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """Handles metrics calculation and monitoring."""
    
    def __init__(self):
        self.request_history = deque(maxlen=1000)  # Keep last 1000 requests
        self.model_performance = defaultdict(list)
        self.error_counts = defaultdict(int)
        self.start_time = datetime.utcnow()
        
    async def log_request(self, request_id: str, model_type: str, 
                         processing_time: float, success: bool):
        """Log a request for metrics calculation."""
        try:
            request_data = {
                'request_id': request_id,
                'model_type': model_type,
                'processing_time': processing_time,
                'success': success,
                'timestamp': datetime.utcnow()
            }
            
            self.request_history.append(request_data)
            self.model_performance[model_type].append({
                'processing_time': processing_time,
                'success': success,
                'timestamp': datetime.utcnow()
            })
            
            if not success:
                self.error_counts[model_type] += 1
            
            logger.debug(f"Logged request {request_id} for metrics")
            
        except Exception as e:
            logger.error(f"Error logging request: {str(e)}")
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics."""
        try:
            current_time = datetime.utcnow()
            
            # Calculate basic metrics
            total_requests = len(self.request_history)
            successful_requests = sum(1 for req in self.request_history if req['success'])
            failed_requests = total_requests - successful_requests
            
            # Calculate average processing time
            processing_times = [req['processing_time'] for req in self.request_history if req['success']]
            avg_processing_time = statistics.mean(processing_times) if processing_times else 0
            
            # Calculate model-specific metrics
            model_metrics = {}
            for model_type, requests in self.model_performance.items():
                if requests:
                    model_processing_times = [req['processing_time'] for req in requests if req['success']]
                    model_success_rate = sum(1 for req in requests if req['success']) / len(requests)
                    
                    model_metrics[model_type] = {
                        'total_requests': len(requests),
                        'success_rate': model_success_rate,
                        'average_processing_time': statistics.mean(model_processing_times) if model_processing_times else 0,
                        'error_count': self.error_counts[model_type]
                    }
            
            # Calculate uptime
            uptime = (current_time - self.start_time).total_seconds()
            
            # Calculate requests per minute
            recent_requests = [
                req for req in self.request_history 
                if (current_time - req['timestamp']).total_seconds() < 60
            ]
            requests_per_minute = len(recent_requests)
            
            return {
                'total_requests': total_requests,
                'successful_requests': successful_requests,
                'failed_requests': failed_requests,
                'success_rate': successful_requests / total_requests if total_requests > 0 else 0,
                'average_processing_time': avg_processing_time,
                'requests_per_minute': requests_per_minute,
                'uptime_seconds': uptime,
                'model_performance': model_metrics,
                'error_counts': dict(self.error_counts),
                'timestamp': current_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {}
    
    async def get_model_performance(self, model_type: str) -> Dict[str, Any]:
        """Get performance metrics for a specific model."""
        try:
            if model_type not in self.model_performance:
                return {}
            
            requests = self.model_performance[model_type]
            if not requests:
                return {}
            
            # Calculate metrics
            total_requests = len(requests)
            successful_requests = sum(1 for req in requests if req['success'])
            failed_requests = total_requests - successful_requests
            
            processing_times = [req['processing_time'] for req in requests if req['success']]
            
            return {
                'model_type': model_type,
                'total_requests': total_requests,
                'successful_requests': successful_requests,
                'failed_requests': failed_requests,
                'success_rate': successful_requests / total_requests,
                'average_processing_time': statistics.mean(processing_times) if processing_times else 0,
                'min_processing_time': min(processing_times) if processing_times else 0,
                'max_processing_time': max(processing_times) if processing_times else 0,
                'error_count': self.error_counts[model_type]
            }
            
        except Exception as e:
            logger.error(f"Error getting model performance: {str(e)}")
            return {}
    
    async def get_recent_requests(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent requests."""
        try:
            recent_requests = list(self.request_history)[-limit:]
            return [
                {
                    'request_id': req['request_id'],
                    'model_type': req['model_type'],
                    'processing_time': req['processing_time'],
                    'success': req['success'],
                    'timestamp': req['timestamp'].isoformat()
                }
                for req in recent_requests
            ]
            
        except Exception as e:
            logger.error(f"Error getting recent requests: {str(e)}")
            return []
    
    async def reset_metrics(self):
        """Reset all metrics."""
        try:
            self.request_history.clear()
            self.model_performance.clear()
            self.error_counts.clear()
            self.start_time = datetime.utcnow()
            logger.info("Metrics reset successfully")
            
        except Exception as e:
            logger.error(f"Error resetting metrics: {str(e)}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status based on metrics."""
        try:
            current_time = datetime.utcnow()
            
            # Check if we have recent requests
            recent_requests = [
                req for req in self.request_history 
                if (current_time - req['timestamp']).total_seconds() < 300  # Last 5 minutes
            ]
            
            # Calculate health score
            if not recent_requests:
                health_score = 0.5  # No recent activity
            else:
                success_rate = sum(1 for req in recent_requests if req['success']) / len(recent_requests)
                avg_processing_time = statistics.mean([req['processing_time'] for req in recent_requests if req['success']])
                
                # Health score based on success rate and processing time
                health_score = success_rate * (1 - min(avg_processing_time / 10, 1))  # Penalize slow processing
            
            return {
                'status': 'healthy' if health_score > 0.8 else 'degraded' if health_score > 0.5 else 'unhealthy',
                'health_score': health_score,
                'recent_requests': len(recent_requests),
                'uptime_seconds': (current_time - self.start_time).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Error getting health status: {str(e)}")
            return {'status': 'unknown', 'health_score': 0}
