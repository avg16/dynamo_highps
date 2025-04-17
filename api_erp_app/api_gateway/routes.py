import logging
import json
from flask import Blueprint, request, jsonify
from api_gateway.auth import token_required
from adapters.adapter_factory import AdapterFactory
from services.dummy_data_mapper import DataMapper

# from services.maintenance_checker import MaintenanceChecker
from services.ai_predictor import AIPredictorService
from services.response_aggregator import ResponseAggregator
from utils.messaging import MessageBroker
from utils.logger import APILogger

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

api_blueprint = Blueprint('api', __name__)

data_mapper = DataMapper()
# maintenance_checker = MaintenanceChecker()
ai_predictor = AIPredictorService()
response_aggregator = ResponseAggregator()
message_broker = MessageBroker()
api_logger = APILogger()

adapter_cache = {}

def get_adapter(system_name):
    if system_name not in adapter_cache:
        adapter_cache[system_name] = AdapterFactory.create_adapter(system_name)
    return adapter_cache[system_name]

@api_blueprint.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "api_gateway",
        "supported_erp_systems": AdapterFactory.get_supported_systems()
    })

@api_blueprint.route('/api/order/<order_id>', methods=['GET'])
@token_required
def get_order(order_id):
    api_logger.log_request(request)
    
    source = request.args.get('source', 'sap').lower()
    
    try:
        adapter = get_adapter(source)
        raw_response = adapter.get_order(order_id)
        normalized_data = data_mapper.transform_response(raw_response, source)

        # maintenance_results = maintenance_checker.check(normalized_data, source)
        message_broker.publish_message(
            'ai_prediction', 
            json.dumps({
                'order_id': order_id,
                'source': source,
                'data': normalized_data
            })
        )

        prediction_results = ai_predictor.predict_compatibility(normalized_data, source)

        final_response = response_aggregator.aggregate(
            normalized_data,
            prediction_results,
        )
        
        # Log the response
        api_logger.log_response(final_response)
        
        return jsonify(final_response)
        
    except Exception as e:
        logger.error(f"Error processing order request: {str(e)}")
        return jsonify({
            "error": "Failed to process request",
            "message": str(e)
        }), 500

@api_blueprint.route('/api/transaction/<transaction_id>', methods=['GET'])
@token_required
def get_transaction(transaction_id):
    api_logger.log_request(request)
    
    # Get source from query params, default to SAP
    source = request.args.get('source', 'sap').lower()
    
    try:
        # Get the appropriate adapter for this ERP system
        adapter = get_adapter(source)
        
        # Get the transaction from the ERP system
        raw_response = adapter.get_transaction(transaction_id)
        
        # Normalize the data
        normalized_data = data_mapper.transform_response(raw_response, source)
        
        # Check for maintenance issues
        # maintenance_results = maintenance_checker.check(normalized_data, source)
        
        message_broker.publish_message(
            'ai_prediction', 
            json.dumps({
                'transaction_id': transaction_id,
                'source': source,
                'data': normalized_data
            })
        )
        
        # Get AI prediction (this would typically be asynchronous but simplified here)
        prediction_results = ai_predictor.predict_compatibility(normalized_data, source)
        
        # Aggregate the response
        final_response = response_aggregator.aggregate(
            normalized_data,
            prediction_results,
        )
        
        # Log the response
        api_logger.log_response(final_response)
        
        return jsonify(final_response)
        
    except Exception as e:
        logger.error(f"Error processing transaction request: {str(e)}")
        return jsonify({
            "error": "Failed to process request",
            "message": str(e)
        }), 500

@api_blueprint.route('/api/metrics', methods=['GET'])
@token_required
def get_metrics():
    """Get system metrics for all active adapters and services"""
    # Get metrics for all active adapters
    adapter_metrics = {}
    for system_name, adapter in adapter_cache.items():
        adapter_metrics[system_name] = adapter.get_health_metrics()
    
    # If no adapters have been used yet, create and get metrics for default ones
    if not adapter_metrics:
        # Add metrics for common ERP systems
        for system in ['sap', 'oracle']:
            adapter = get_adapter(system)
            adapter_metrics[system] = adapter.get_health_metrics()
    
    return jsonify({
        "adapters": adapter_metrics,
        "services": {
            "data_mapper": data_mapper.get_metrics(),
            # "maintenance": maintenance_checker.get_metrics(),
            "ai_predictor": ai_predictor.get_metrics()
        },
        "supported_systems": AdapterFactory.get_supported_systems()
    })

@api_blueprint.route('/api/ai/retrain', methods=['POST'])
@token_required
def retrain_ai_models():
    """Manually trigger AI model retraining"""
    try:
        result = ai_predictor.retrain_models()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error retraining AI models: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Failed to retrain models: {str(e)}"
        }), 500

@api_blueprint.route('/api/ai/info', methods=['GET'])
@token_required
def get_ai_info():
    try:
        metrics = ai_predictor.get_metrics()
        model_info = {
            "model_info": {
                "version": metrics.get("model_version", "unknown"),
                "accuracy": metrics.get("model_accuracy", "unknown"),
                "training_data_points": metrics.get("training_data_points", 0),
                "predictions_made": metrics.get("predictions_made", 0),
                "features_used": [
                    "version_factor - How recent the API version is",
                    "field_coverage - Percentage of required fields present",
                    "time_factor - Time since last API update relative to average update frequency"
                ],
                "model_types": [
                    "Classification model (Random Forest) - Predicts risk level (low/medium/high)",
                    "Regression model (Gradient Boosting) - Predicts continuous risk score"
                ],
                "training_method": "Supervised learning with continuous model improvement"
            }
        }
        return jsonify(model_info)
    except Exception as e:
        logger.error(f"Error getting AI model info: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Failed to get model info: {str(e)}"
        }), 500
