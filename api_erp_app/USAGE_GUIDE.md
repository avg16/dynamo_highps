# ERP Integration Platform - Usage Guide

This guide provides detailed instructions for running, testing, and using the ERP Integration Platform.

## Running the Application

The application is configured to run with Gunicorn. To start the server:

```bash
# Start the application with reloading enabled
gunicorn --bind 0.0.0.0:8000 --reuse-port --reload main:app
```

Alternatively, you can use the provided workflow:

```bash
start workflow "Start application"
```

## Testing API Endpoints

You can use curl to test the API endpoints:

### Authentication

```bash
# Get an authentication token
curl -X POST \
  http://localhost:8000/api/auth/token \
  -H 'Content-Type: application/json' \
  -d '{"username": "admin", "password": "admin"}'
```

The response will include a JWT token that you should use for subsequent requests:

```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": "admin-user-id",
    "username": "admin",
    "roles": ["admin"]
  }
}
```

### Retrieving Order Information

```bash
curl -X GET \
  'http://localhost:5000/api/order/12345?source=sap' \
  -H 'Authorization: Bearer YOUR_TOKEN_HERE'

curl -X GET \
  'http://localhost:5000/api/order/12345?source=oracle' \
  -H 'Authorization: Bearer YOUR_TOKEN_HERE'

# Get order information from a custom ERP system
curl -X GET \
  'http://localhost:5000/api/order/12345?source=netsuite' \
  -H 'Authorization: Bearer YOUR_TOKEN_HERE'
```

### Retrieving Transaction Information

```bash
# Get transaction information from SAP ERP system
curl -X GET \
  'http://localhost:5000/api/transaction/T12345?source=sap' \
  -H 'Authorization: Bearer YOUR_TOKEN_HERE'

# Get transaction information from Oracle ERP system
curl -X GET \
  'http://localhost:5000/api/transaction/T12345?source=oracle' \
  -H 'Authorization: Bearer YOUR_TOKEN_HERE'
```

### Health and Metrics

```bash
# Get health status
curl -X GET http://localhost:5000/api/health

# Get system metrics
curl -X GET \
  http://localhost:5000/api/metrics \
  -H 'Authorization: Bearer YOUR_TOKEN_HERE'

# Get AI model information
curl -X GET \
  http://localhost:5000/api/ai/info \
  -H 'Authorization: Bearer YOUR_TOKEN_HERE'
```

### AI Model Retraining

```bash
# Manually trigger AI model retraining
curl -X POST \
  http://localhost:5000/api/ai/retrain \
  -H 'Authorization: Bearer YOUR_TOKEN_HERE'
```

## Response Format

The API returns responses in a unified format with standardized data and diagnostic information:

```json
{
  "data": {
    // Normalized data from the ERP system
    "orderId": "12345",
    "customerInfo": { ... },
    "items": [ ... ],
    "shippingDetails": { ... }
  },
  "diagnostics": {
    "source": "sap",
    "maintenance": {
      // Maintenance check results
      "requiredFields": { "status": "complete" },
      "version": { "status": "current", "value": "2.1" },
      "dataConsistency": { "status": "valid" }
    },
    "prediction": {
      // AI prediction results
      "compatibilityRisk": "low",
      "confidenceScore": 0.95,
      "suggestedActions": [ ... ]
    },
    "metrics": {
      // Performance metrics
      "responseTime": 0.242,
      "processingTime": 0.156
    }
  }
}
```

## Common Usage Patterns

### Multi-ERP Integration

To integrate with multiple ERP systems simultaneously:

1. Configure the necessary API credentials for each ERP system using environment variables
2. Make parallel requests to each ERP system for the same entity
3. Compare and merge the results as needed

Example:

```javascript
// Example client-side code (Node.js)
async function getOrderFromAllSystems(orderId) {
  const systems = ['sap', 'oracle', 'netsuite'];
  const results = {};
  
  await Promise.all(systems.map(async (system) => {
    const response = await fetch(`http://localhost:5000/api/order/${orderId}?source=${system}`, {
      headers: { 'Authorization': `Bearer ${token}` }
    });
    results[system] = await response.json();
  }));
  
  return results;
}
```

### Adding a Custom ERP System

The platform supports custom ERP systems out of the box. Simply pass the appropriate source parameter:

```bash
curl -X GET \
  'http://localhost:5000/api/order/12345?source=custom_erp' \
  -H 'Authorization: Bearer YOUR_TOKEN_HERE'
```

For better integration, you can add custom formats by setting environment variables or modifying the code:

1. Create a format definition for your system in `adapters/erp_adapter.py`
2. Add environment variable configuration in `adapters/adapter_factory.py`

### Handling Large Data Volumes

For large data volumes, consider:

1. Implementing pagination in your requests
2. Using appropriate filtering to limit result sets
3. Processing data in batches

Example:

```bash
# Get paginated results with limit and offset
curl -X GET \
  'http://localhost:5000/api/order/12345?source=sap&limit=10&offset=0' \
  -H 'Authorization: Bearer YOUR_TOKEN_HERE'
```

## Troubleshooting

### Common Issues

1. **Authentication Failures**
   - Ensure the token is valid and not expired
   - Check that you're including the 'Bearer ' prefix in the Authorization header

2. **Unknown ERP Source**
   - Verify the ERP system name is passed correctly in the 'source' parameter
   - Confirm that you've configured any necessary environment variables for that ERP system

3. **Rate Limiting**
   - If you receive a 429 response, you've exceeded the rate limit
   - Implement backoff and retry logic in your client

4. **Circuit Breaker Tripped**
   - If an ERP system is experiencing issues, the circuit breaker may open
   - The API will return a 503 response in this case
   - Wait for the circuit to reset or use a different ERP system

### Viewing Logs

```bash
# View application logs
tail -f /logs/app.log

# View specific service logs
grep "ai_predictor" /logs/app.log
```

## Best Practices

1. **Always Use Authentication**
   - Secure all API requests with the JWT token
   - Refresh tokens before they expire

2. **Handle Error Responses Gracefully**
   - Implement proper error handling in your client
   - Check response status codes and error messages

3. **Monitor Metrics and Diagnostics**
   - Regularly check the metrics endpoint for system health
   - Review prediction and maintenance diagnostics

4. **Keep ERP API Credentials Secure**
   - Use environment variables for API credentials
   - Never hardcode sensitive information in your code

5. **Leverage AI Predictions**
   - Use the prediction results to proactively address potential issues
   - Follow suggested actions to improve compatibility

6. **Stay Current with ERP Versions**
   - Monitor version deprecation warnings
   - Update to newer versions before sunset dates

## Advanced Configuration

### Configuring Additional ERP Systems

To add support for additional ERP systems:

1. Define the response format in `RESPONSE_FORMATS` in `adapters/erp_adapter.py`
2. Add environment variable mappings in `DEFAULT_ENV_VARS` in `adapters/adapter_factory.py`
3. Configure the appropriate environment variables

### Customizing Field Mapping

To customize field mapping between ERP systems:

1. Use the Field Discovery service to analyze responses
2. Generate mapping suggestions
3. Apply custom mappings in your application logic

### Setting Up Redis for Enhanced Rate Limiting

For production use, configure Redis for more robust rate limiting:

1. Install Redis server
2. Set the `REDIS_HOST`, `REDIS_PORT`, and `REDIS_PASSWORD` environment variables
3. Restart the application

### Setting Up RabbitMQ for Enhanced Message Queuing

For production use, configure RabbitMQ for robust message queuing:

1. Install RabbitMQ server
2. Set the `RABBITMQ_HOST`, `RABBITMQ_PORT`, `RABBITMQ_USER`, and `RABBITMQ_PASSWORD` environment variables
3. Restart the application