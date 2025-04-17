# ERP Integration Platform

A comprehensive middleware solution for integrating with diverse ERP systems. This platform provides a unified API gateway, intelligent data mapping, and predictive maintenance for seamless ERP system connectivity.

## Features

- **Unified API Gateway**: Secure, rate-limited access to multiple ERP systems
- **Intelligent Adapters**: Support for SAP, Oracle, NetSuite, Dynamics, and custom ERP systems
- **Data Normalization**: Convert diverse ERP data formats into standardized schemas
- **Circuit Breaker Pattern**: Built-in resilience for handling ERP system failures
- **AI-Powered Predictions**: Predictive analytics for API compatibility and maintenance
- **Comprehensive Security**: Advanced authentication, encryption, and monitoring
- **Automated Documentation**: Self-documenting API with OpenAPI specification
- **Automated Field Discovery**: Dynamic schema mapping across different ERP systems
- **Version Management**: API version compatibility, deprecation warnings, and update recommendations

## Getting Started

### Prerequisites

- Python 3.11+
- Redis (optional, for enhanced rate limiting)
- RabbitMQ (optional, for enhanced message queuing)
- PostgreSQL (optional, for persistent storage)

### Installation

1. Clone the repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Configure environment variables (see Configuration section)
4. Start the application:

```bash
gunicorn --bind 0.0.0.0:5000 --reuse-port --reload main:app
```

## Configuration

The platform is configured using environment variables:

### Core Settings

- `API_HOST`: Host to bind the API server (default: "0.0.0.0")
- `API_PORT`: Port to bind the API server (default: 5000)
- `API_DEBUG`: Enable debug mode (default: "true")
- `SECRET_KEY`: Secret key for encryption (default: "dev-secret-key")
- `JWT_SECRET`: Secret for JWT tokens (default: "dev-jwt-secret")
- `JWT_EXPIRATION`: JWT token expiration in seconds (default: 3600)

### Rate Limiting

- `RATE_LIMIT`: Maximum number of requests per window (default: 10)
- `RATE_LIMIT_WINDOW`: Rate limit window in seconds (default: 60)

### ERP System Connections

- `SAP_API_URL`: SAP API base URL
- `SAP_API_KEY`: SAP API authentication key
- `ORACLE_API_URL`: Oracle API base URL
- `ORACLE_API_KEY`: Oracle API authentication key
- `NETSUITE_API_URL`: NetSuite API base URL
- `NETSUITE_API_KEY`: NetSuite API authentication key
- `DYNAMICS_API_URL`: Dynamics API base URL
- `DYNAMICS_API_KEY`: Dynamics API authentication key

### Circuit Breaker Configuration

- `CIRCUIT_MAX_FAILURES`: Maximum failures before opening circuit (default: 3)
- `CIRCUIT_RESET_TIMEOUT`: Reset timeout in seconds (default: 30)

### Redis Configuration (Optional)

- `REDIS_HOST`: Redis host (default: "localhost")
- `REDIS_PORT`: Redis port (default: 6379)
- `REDIS_PASSWORD`: Redis password

### RabbitMQ Configuration (Optional)

- `RABBITMQ_HOST`: RabbitMQ host (default: "localhost")
- `RABBITMQ_PORT`: RabbitMQ port (default: 5672)
- `RABBITMQ_USER`: RabbitMQ username (default: "guest")
- `RABBITMQ_PASSWORD`: RabbitMQ password (default: "guest")

### Logging

- `LOG_LEVEL`: Logging level (default: "DEBUG")

## API Documentation

The API Gateway provides the following primary endpoints:

### Authentication

- `POST /api/auth/token`: Get a JWT authentication token

### Order Operations

- `GET /api/order/{order_id}?source={erp_system}`: Get order information from specified ERP system

### Transaction Operations

- `GET /api/transaction/{transaction_id}?source={erp_system}`: Get transaction information from specified ERP system

### Health & Metrics

- `GET /api/health`: Health check endpoint
- `GET /api/metrics`: Get system metrics
- `GET /api/ai/info`: Get information about the AI prediction models

### Administration

- `POST /api/ai/retrain`: Manually trigger AI model retraining

## Architecture

The platform consists of the following core components:

1. **API Gateway**: Entry point for all requests, handles authentication, rate limiting, and routing
2. **Adapter Factory**: Creates appropriate adapters for different ERP systems
3. **Data Mapper**: Transforms ERP-specific data formats into a standardized schema
4. **Maintenance Checker**: Verifies version compliance, key schema fields, and data consistency
5. **AI Predictor**: Predicts compatibility risks for ERP data
6. **Response Aggregator**: Combines data from various services into a unified response

## Advanced Features

### Adapter Factory

The Adapter Factory allows easy integration with multiple ERP systems:

```python
from adapters.adapter_factory import AdapterFactory

# Create an adapter for SAP
sap_adapter = AdapterFactory.create_adapter('sap')

# Create an adapter for Oracle
oracle_adapter = AdapterFactory.create_adapter('oracle')

# Create an adapter for a custom ERP system
custom_adapter = AdapterFactory.create_adapter('custom', {
    'api_url': 'https://api.custom-erp.example.com',
    'api_key': 'YOUR_API_KEY'
})
```

### Field Discovery

The Field Discovery service automatically analyzes ERP responses to identify fields and their types:

```python
from services.field_discovery import FieldDiscovery

field_discovery = FieldDiscovery()

# Analyze a response from an ERP system
analysis = field_discovery.analyze_response('sap', 'order', response_data)

# Generate mapping suggestions
mappings = field_discovery.generate_mapping_suggestions('order')

# Generate a unified schema
schema = field_discovery.generate_schema('order')
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.