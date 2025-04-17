import logging
import datetime
import json
import os
import threading
import time
import uuid

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class APILogger:
    """
    Logger for API requests and responses
    Records request/response details and provides metrics
    """
    
    def __init__(self):
        self.requests = []
        self.responses = []
        self.max_log_entries = 1000  # Maximum number of entries to keep in memory
        self.request_count = 0
        self.error_count = 0
        self.start_time = datetime.datetime.now()
        
        # Log retention period (24 hours)
        self.retention_period = 24 * 60 * 60  # seconds
        
        # Start log cleanup thread
        cleanup_thread = threading.Thread(target=self._cleanup_logs, daemon=True)
        cleanup_thread.start()
    
    def log_request(self, request):
        """
        Log an API request
        
        Args:
            request: The Flask request object
        """
        self.request_count += 1
        
        # Create request log entry
        request_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        
        entry = {
            'request_id': request_id,
            'timestamp': timestamp,
            'method': request.method,
            'path': request.path,
            'remote_addr': request.remote_addr,
            'args': dict(request.args),
            'headers': {k: v for k, v in request.headers.items() if k.lower() != 'authorization'},
            'content_length': request.content_length
        }
        
        # Add entry to log
        self.requests.append(entry)
        
        # Trim log if necessary
        if len(self.requests) > self.max_log_entries:
            self.requests = self.requests[-self.max_log_entries:]
            
        logger.debug(f"Logged API request: {request.method} {request.path}")
        return request_id
    
    def log_response(self, response, request_id=None, error=None):
        """
        Log an API response
        
        Args:
            response: The response data
            request_id: The ID of the corresponding request (optional)
            error: Error information if the request failed (optional)
        """
        # Create response log entry
        timestamp = datetime.datetime.now().isoformat()
        
        # Check if response has an error
        has_error = error is not None
        if not has_error and isinstance(response, dict):
            has_error = 'error' in response
            
        if has_error:
            self.error_count += 1
            
        entry = {
            'timestamp': timestamp,
            'request_id': request_id,
            'has_error': has_error,
            'error': error,
            'response_size': len(json.dumps(response)) if isinstance(response, (dict, list)) else 0
        }
        
        # Add entry to log
        self.responses.append(entry)
        
        # Trim log if necessary
        if len(self.responses) > self.max_log_entries:
            self.responses = self.responses[-self.max_log_entries:]
            
        logger.debug(f"Logged API response for request {request_id}")
    
    def get_metrics(self):
        """Get metrics from the logger"""
        current_time = datetime.datetime.now()
        uptime_seconds = (current_time - self.start_time).total_seconds()
        
        return {
            'request_count': self.request_count,
            'error_count': self.error_count,
            'error_rate': round(self.error_count / max(self.request_count, 1) * 100, 2),
            'uptime_seconds': int(uptime_seconds),
            'requests_per_minute': round(self.request_count / max(uptime_seconds / 60, 1), 2)
        }
    
    def _cleanup_logs(self):
        """Periodically clean up old log entries"""
        while True:
            try:
                current_time = datetime.datetime.now()
                cutoff = current_time - datetime.timedelta(seconds=self.retention_period)
                cutoff_str = cutoff.isoformat()
                
                # Filter out old entries
                self.requests = [r for r in self.requests if r.get('timestamp', '') >= cutoff_str]
                self.responses = [r for r in self.responses if r.get('timestamp', '') >= cutoff_str]
                
                logger.debug(f"Cleaned up logs older than {cutoff}")
            except Exception as e:
                logger.error(f"Error cleaning up logs: {str(e)}")
                
            # Sleep for 1 hour before next cleanup
            time.sleep(3600)
