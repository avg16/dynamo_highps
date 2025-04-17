import logging
import os
from adapters.erp_adapter import ERPAdapter

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AdapterFactory:
    """
    Factory class for creating ERP adapters
    Maintains a registry of adapter classes for different ERP systems
    """
    
    # Registry of default environment variable names for different ERP systems
    DEFAULT_ENV_VARS = {
        'sap': {
            'api_url': 'SAP_API_URL',
            'api_key': 'SAP_API_KEY'
        },
        'oracle': {
            'api_url': 'ORACLE_API_URL',
            'api_key': 'ORACLE_API_KEY'
        },
        'netsuite': {
            'api_url': 'NETSUITE_API_URL',
            'api_key': 'NETSUITE_API_KEY'
        },
        'dynamics': {
            'api_url': 'DYNAMICS_API_URL',
            'api_key': 'DYNAMICS_API_KEY'
        }
    }
    
    @classmethod
    def create_adapter(cls, system_name, config=None):
        """
        Create an adapter for the specified ERP system
        
        Args:
            system_name: The name of the ERP system
            config: Optional configuration dictionary
            
        Returns:
            An instance of the appropriate adapter
        """
        # Normalize the system name to lowercase
        system_name = system_name.lower()
        
        # If no config is provided, try to load from environment variables
        if not config:
            config = cls._load_config_from_env(system_name)
        
        # Create and return the generic adapter with the appropriate configuration
        logger.info(f"Creating adapter for {system_name} ERP system")
        return ERPAdapter(system_name, config)
    
    @classmethod
    def _load_config_from_env(cls, system_name):
        """
        Load configuration from environment variables
        
        Args:
            system_name: The name of the ERP system
            
        Returns:
            Configuration dictionary
        """
        config = {}
        
        # Look up the standard environment variable names for this ERP system
        env_vars = cls.DEFAULT_ENV_VARS.get(system_name, {})
        
        # Load API URL
        api_url_env = env_vars.get('api_url')
        if api_url_env and os.environ.get(api_url_env):
            config['api_url'] = os.environ.get(api_url_env)
            
        # Load API key
        api_key_env = env_vars.get('api_key')
        if api_key_env and os.environ.get(api_key_env):
            config['api_key'] = os.environ.get(api_key_env)
            
        # If there's a generic config for this system, use it as a fallback
        if not config:
            generic_url_env = f"{system_name.upper()}_API_URL"
            generic_key_env = f"{system_name.upper()}_API_KEY"
            
            if os.environ.get(generic_url_env):
                config['api_url'] = os.environ.get(generic_url_env)
            
            if os.environ.get(generic_key_env):
                config['api_key'] = os.environ.get(generic_key_env)
        
        return config
        
    @classmethod
    def get_supported_systems(cls):
        """
        Get a list of supported ERP systems
        
        Returns:
            List of supported ERP system names
        """
        return list(cls.DEFAULT_ENV_VARS.keys()) + ['custom']