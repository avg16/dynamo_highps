import time
import logging
from flask import request, jsonify
import redis
import os

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

redis_host = os.environ.get("REDIS_HOST", "localhost")
redis_port = int(os.environ.get("REDIS_PORT", 6379))
redis_password = os.environ.get("REDIS_PASSWORD", None)

try:
    redis_client = redis.Redis(
        host=redis_host, 
        port=redis_port,
        password=redis_password,
        socket_connect_timeout=1,
        decode_responses=True
    )
    redis_client.ping()  # Test connection
    use_redis = True
    logger.info("Using Redis for rate limiting")
except (redis.ConnectionError, redis.exceptions.TimeoutError):
    use_redis = False
    in_memory_limits = {}
    logger.warning("Redis not available, using in-memory rate limiting")

RATE_LIMIT = 10  # requests
TIME_WINDOW = 60  # seconds

def get_rate_limit_key(request):
    """Get key for rate limiting based on IP and endpoint"""
    return f"rate_limit:{request.remote_addr}:{request.endpoint}"

def is_rate_limited(key):
    """Check if request is rate limited"""
    current_time = int(time.time())
    window_start = current_time - TIME_WINDOW
    
    if key not in in_memory_limits:
            in_memory_limits[key] = []
        
    in_memory_limits[key] = [ts for ts in in_memory_limits[key] if ts > window_start]
    in_memory_limits[key].append(current_time)
    request_count = len(in_memory_limits[key])
    return request_count > RATE_LIMIT

def rate_limit_middleware():
    key = get_rate_limit_key(request)
    
    if is_rate_limited(key):
        logger.warning(f"Rate limit exceeded for {key}")
        return jsonify({
            "error": "Rate limit exceeded",
            "message": f"Too many requests. Maximum {RATE_LIMIT} requests per {TIME_WINDOW} seconds."
        }), 429
    
    return None

def configure_rate_limiting(app):
    """Configure rate limiting for the app"""
    @app.before_request
    def check_rate_limit():
        if not request.path.startswith('/api/'):
            return None
        return rate_limit_middleware()
