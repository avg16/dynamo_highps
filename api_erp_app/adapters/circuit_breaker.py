import logging
import time
import functools
import threading

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class CircuitBreakerState:
    """
    Circuit breaker state management
    Tracks failure counts and circuit state (open, closed, half-open)
    """
    CLOSED = 'closed'  # Normal operation
    OPEN = 'open'      # Circuit breaker tripped, failing fast
    HALF_OPEN = 'half-open'  # Testing if service is back online
    
    def __init__(self, max_failures, reset_timeout):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.state = self.CLOSED
        self.last_failure_time = 0
        self.lock = threading.RLock()
        
    def record_success(self):
        """Record a successful call"""
        with self.lock:
            self.failures = 0
            self.state = self.CLOSED
            
    def record_failure(self):
        """Record a failed call"""
        with self.lock:
            self.failures += 1
            self.last_failure_time = time.time()
            
            if self.failures >= self.max_failures:
                self.state = self.OPEN
                logger.warning(f"Circuit breaker opened after {self.failures} failures")
    
    def can_attempt_request(self):
        """Check if a request can be attempted based on current state"""
        with self.lock:
            if self.state == self.CLOSED:
                return True
                
            if self.state == self.OPEN:
                # Check if enough time has passed to try a request
                timeout_expired = time.time() > self.last_failure_time + self.reset_timeout
                if timeout_expired:
                    logger.info("Circuit breaker entering half-open state")
                    self.state = self.HALF_OPEN
                    return True
                return False
                
            # Half-open state allows one request to test the service
            return True

# Global dictionary to store circuit breaker states for different functions
circuit_breakers = {}

def circuit_breaker(max_failures=3, reset_timeout=30):
    """
    Circuit breaker decorator
    Wraps a function to implement the circuit breaker pattern
    
    Args:
        max_failures: Maximum number of failures before opening the circuit
        reset_timeout: Time in seconds before attempting to close the circuit again
    """
    def decorator(func):
        # Create a unique key for this function
        func_key = f"{func.__module__}.{func.__qualname__}"
        
        # Create a circuit breaker state for this function if it doesn't exist
        if func_key not in circuit_breakers:
            circuit_breakers[func_key] = CircuitBreakerState(max_failures, reset_timeout)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            breaker = circuit_breakers[func_key]
            
            if not breaker.can_attempt_request():
                logger.warning(f"Circuit breaker open for {func_key}, failing fast")
                # If the first argument is self, update its circuit_trips attribute if it exists
                if args and hasattr(args[0], 'circuit_trips'):
                    args[0].circuit_trips += 1
                raise Exception("Circuit breaker open, service unavailable")
            
            try:
                result = func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                # If the first argument is self, update its circuit_trips attribute if it exists
                if args and hasattr(args[0], 'circuit_trips'):
                    args[0].circuit_trips += 1
                logger.error(f"Circuit breaker: function {func_key} failed with: {str(e)}")
                raise
                
        return wrapper
    return decorator
