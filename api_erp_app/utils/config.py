import os
import logging
import json

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Config:
    """
    Configuration management for the ERP Integration Platform
    Loads configuration from environment variables with fallbacks
    """
    
    # API Gateway configuration
    API_HOST = os.environ.get("API_HOST", "0.0.0.0")
    API_PORT = int(os.environ.get("API_PORT", 5000))
    API_DEBUG = os.environ.get("API_DEBUG", "true").lower() == "true"
    
    # Security configuration
    JWT_SECRET = os.environ.get("JWT_SECRET", "dev-jwt-secret")
    JWT_EXPIRATION = int(os.environ.get("JWT_EXPIRATION", 3600))  # 1 hour
    
    # Rate limiting configuration
    RATE_LIMIT = int(os.environ.get("RATE_LIMIT", 10))
    RATE_LIMIT_WINDOW = int(os.environ.get("RATE_LIMIT_WINDOW", 60))  # seconds
    
    # SAP Adapter configuration
    # SAP_API_URL = os.environ.get("SAP_API_URL", "https://api.sap.example.com")
    # SAP_API_KEY = os.environ.get("SAP_API_KEY", "sample-key")
    
    # Oracle Adapter configuration
    # ORACLE_API_URL = os.environ.get("ORACLE_API_URL", "https://api.oracle.example.com")
    # ORACLE_API_KEY = os.environ.get("ORACLE_API_KEY", "sample-key")
    
    CIRCUIT_MAX_FAILURES = int(os.environ.get("CIRCUIT_MAX_FAILURES", 3))
    CIRCUIT_RESET_TIMEOUT = int(os.environ.get("CIRCUIT_RESET_TIMEOUT", 30))  # seconds
    
    REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
    REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", None)
    
    RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "localhost")
    RABBITMQ_PORT = int(os.environ.get("RABBITMQ_PORT", 5672))
    RABBITMQ_USER = os.environ.get("RABBITMQ_USER", "guest")
    RABBITMQ_PASSWORD = os.environ.get("RABBITMQ_PASSWORD", "guest")
    
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "DEBUG")
    
    @classmethod
    def get_all(cls):
        """Get all configuration values (excluding sensitive information)"""
        config_dict = {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('__') and not callable(getattr(cls, key))
            and key not in ['JWT_SECRET', 'REDIS_PASSWORD', 'RABBITMQ_PASSWORD']
        }
        return config_dict
    
    @classmethod
    def load_from_file(cls, file_path):
        """
        Load configuration from a JSON file
        
        Args:
            file_path: Path to the configuration file
        """
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Configuration file not found: {file_path}")
                return
                
            with open(file_path, 'r') as f:
                config_data = json.load(f)
                
            # Update configuration values
            for key, value in config_data.items():
                if hasattr(cls, key):
                    setattr(cls, key, value)
                    logger.debug(f"Loaded configuration: {key}={value}")
                    
            logger.info(f"Loaded configuration from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration from {file_path}: {str(e)}")
