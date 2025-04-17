import logging
import time
import json
import random
import os
from abc import ABC, abstractmethod
from adapters.circuit_breaker import circuit_breaker

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class BaseAdapter(ABC):
    """
    Base adapter class to define the interface for all ERP adapters
    All specific ERP adapters should inherit from this class
    """
    
    def __init__(self, system_name, config=None):
        """
        Initialize the adapter with system name and optional config
        
        Args:
            system_name: The name of the ERP system
            config: Optional configuration dictionary
        """
        self.system_name = system_name
        self.config = config or {}
        self.base_url = self.config.get('api_url', f"https://api.{system_name}.example.com")
        self.api_key = self.config.get('api_key', "sample-key")
        self.timeout = self.config.get('timeout', 10)  # seconds
        self.circuit_trips = 0
        self.request_count = 0
        self.error_count = 0
        self.response_times = []
    
    @abstractmethod
    def get_order(self, order_id):
        """
        Get order details from the ERP system
        
        Args:
            order_id: The ID of the order to retrieve
            
        Returns:
            Dictionary containing order details in the ERP's native format
        """
        pass
    
    @abstractmethod
    def get_transaction(self, transaction_id):
        """
        Get transaction details from the ERP system
        
        Args:
            transaction_id: The ID of the transaction to retrieve
            
        Returns:
            Dictionary containing transaction details in the ERP's native format
        """
        pass
    
    def get_health_metrics(self):
        """
        Get health metrics for the adapter
        
        Returns:
            Dictionary with health metrics
        """
        avg_response_time = sum(self.response_times[-100:]) / max(len(self.response_times[-100:]), 1)
        
        return {
            "system": self.system_name,
            "status": "healthy" if self.circuit_trips < 3 else "degraded",
            "requests": self.request_count,
            "errors": self.error_count,
            "circuit_trips": self.circuit_trips,
            "avg_response_time": round(avg_response_time, 3) if self.response_times else 0,
            "error_rate": round(self.error_count / max(self.request_count, 1) * 100, 2)
        }
    
    def _simulate_api_call(self, endpoint, entity_id, simulate_func, latency=0.2, failure_rate=0.05):
        """
        Helper method to simulate an API call with proper metrics
        
        Args:
            endpoint: The API endpoint being called
            entity_id: The ID of the entity being requested
            simulate_func: Function to generate simulated response
            latency: Simulated latency in seconds
            failure_rate: Probability of failure (0-1)
            
        Returns:
            Simulated API response
            
        Raises:
            Exception: If the simulated API call fails
        """
        self.request_count += 1
        
        # Simulate API call
        logger.info(f"Calling {self.system_name} API for {endpoint} {entity_id}")
        start_time = time.time()
        
        try:
            # Simulate network latency
            time.sleep(latency)
            
            # Simulate occasional failure
            if random.random() < failure_rate:
                self.error_count += 1
                raise Exception(f"{self.system_name} API temporarily unavailable")
            
            # Get a simulated response
            response = simulate_func(entity_id)
            
            # Record response time
            elapsed = time.time() - start_time
            self.response_times.append(elapsed)
            
            return response
            
        except Exception as e:
            logger.error(f"Error calling {self.system_name} API: {str(e)}")
            self.error_count += 1
            raise