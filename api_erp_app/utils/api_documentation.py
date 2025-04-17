import logging
import json
import inspect
import os
from flask import current_app

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class APIDocumentation:
    def __init__(self):
        self.endpoints = {}
        self.schemas = {}
        
    def register_endpoint(self, route, method, description, params=None, responses=None):
        if route not in self.endpoints:
            self.endpoints[route] = {}
            
        self.endpoints[route][method] = {
            'description': description,
            'parameters': params or {},
            'responses': responses or {'200': 'Success'}
        }
        
        logger.debug(f"Registered documentation for {method} {route}")
        
    def register_schema(self, name, schema):
        """
        Register a data schema for documentation
        
        Args:
            name: Name of the schema
            schema: Dictionary defining the schema
        """
        self.schemas[name] = schema
        logger.debug(f"Registered schema: {name}")
        
    def auto_document_blueprint(self, blueprint):
        """
        Automatically document all routes in a Flask blueprint
        
        Args:
            blueprint: The Flask blueprint to document
        """
        for rule in current_app.url_map.iter_rules():
            if rule.endpoint.startswith(blueprint.name):
                # Get the view function
                view_func = current_app.view_functions[rule.endpoint]
                
                # Extract method and description from docstring
                if view_func.__doc__:
                    description = view_func.__doc__.strip()
                else:
                    description = "No description available"
                    
                # Convert route from Flask format to documentation format
                route = str(rule)
                route = route.replace('<', '{').replace('>', '}')
                
                # Extract parameters from route
                params = {}
                for arg in rule.arguments:
                    params[arg] = {
                        'type': 'string',
                        'description': f'The {arg} parameter',
                        'required': True
                    }
                    
                # Add query parameters if mentioned in docstring
                if 'Query params:' in description:
                    query_section = description.split('Query params:')[1].strip()
                    for line in query_section.split('\n'):
                        if '-' in line:
                            param_name = line.split('-')[1].strip().split(':')[0].strip()
                            param_desc = line.split(':')[1].strip() if ':' in line else ''
                            params[param_name] = {
                                'type': 'string',
                                'description': param_desc,
                                'in': 'query',
                                'required': False
                            }
                
                # Register the endpoint
                for method in rule.methods:
                    if method not in ['HEAD', 'OPTIONS']:
                        self.register_endpoint(
                            route, 
                            method, 
                            description,
                            params
                        )
        
    def generate_openapi_spec(self, title, version, description=None):

        paths = {}
        
        for route, methods in self.endpoints.items():
            paths[route] = {}
            
            for method, details in methods.items():
                path_item = {
                    'summary': details['description'].split('\n')[0],
                    'description': details['description'],
                    'responses': {}
                }
                
                if details['parameters']:
                    path_item['parameters'] = []
                    for param_name, param_details in details['parameters'].items():
                        param_in = param_details.get('in', 'path' if '{' + param_name + '}' in route else 'query')
                        path_item['parameters'].append({
                            'name': param_name,
                            'in': param_in,
                            'description': param_details.get('description', ''),
                            'required': param_details.get('required', True),
                            'schema': {
                                'type': param_details.get('type', 'string')
                            }
                        })
                
                for status_code, response_desc in details['responses'].items():
                    path_item['responses'][status_code] = {
                        'description': response_desc
                    }
                    
                paths[route][method.lower()] = path_item
        spec = {
            'openapi': '3.0.0',
            'info': {
                'title': title,
                'version': version,
                'description': description or ''
            },
            'paths': paths,
            'components': {
                'schemas': self.schemas
            }
        }
        
        return spec
        
    def save_openapi_spec(self, filename, title, version, description=None):
        """
        Generate and save OpenAPI specification document to a file
        
        Args:
            filename: Output filename
            title: API title
            version: API version
            description: API description
        """
        spec = self.generate_openapi_spec(title, version, description)
        
        with open(filename, 'w') as f:
            json.dump(spec, f, indent=2)
            
        logger.info(f"Saved OpenAPI specification to {filename}")
        return filename