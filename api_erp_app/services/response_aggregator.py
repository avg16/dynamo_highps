import logging
import datetime
import json

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ResponseAggregator:
    def __init__(self):
        self.responses_aggregated = 0
        self.last_aggregation_time = None
        
    def aggregate(self, normalized_data, maintenance_results, prediction_results):

        self.responses_aggregated += 1
        self.last_aggregation_time = datetime.datetime.now().isoformat()
        
        logger.info("Aggregating response from multiple services")
        response = {
            "data": self._prepare_data(normalized_data),
            "diagnostics": {
                "maintenance": self._prepare_maintenance(maintenance_results),
                "prediction": self._prepare_prediction(prediction_results),
                "system_info": self._get_system_info(normalized_data)
            },
            "meta": {
                "generated_at": self.last_aggregation_time,
                "version": "1.0"
            }
        }
        response["diagnostics"]["overall_health"] = self._calculate_overall_health(
            maintenance_results,
            prediction_results
        )
        
        return response
        
    def _prepare_data(self, normalized_data):
        if not normalized_data:
            return {}
            
        data_copy = normalized_data.copy() if normalized_data else {}
        if '_meta' in data_copy:
            del data_copy['_meta']
            
        return data_copy
        
    def _prepare_maintenance(self, maintenance_results):
        if not maintenance_results:
            return {
                "status": "unknown",
                "issues": [],
                "recommendations": []
            }
            
        return {
            "status": maintenance_results.get("status", "unknown"),
            "issues": maintenance_results.get("issues", []),
            "recommendations": maintenance_results.get("recommendations", []),
            "checked_at": maintenance_results.get("checked_at", self.last_aggregation_time)
        }
        
    def _prepare_prediction(self, prediction_results):
        if not prediction_results:
            return {
                "risk_level": "unknown",
                "risk_score": 0.5,
                "message": "No prediction available",
                "suggestions": []
            }
        return {
            "risk_level": prediction_results.get("risk_level", "unknown"),
            "risk_score": prediction_results.get("risk_score", 0.5),
            "message": prediction_results.get("message", ""),
            "suggestions": prediction_results.get("suggestions", []),
            "predicted_at": prediction_results.get("predicted_at", self.last_aggregation_time)
        }
        
    def _get_system_info(self, normalized_data):
        source = normalized_data.get('_meta', {}).get('source', 'unknown')
        entity_type = normalized_data.get('_meta', {}).get('entity_type', 'unknown')
        
        return {
            "source_system": source,
            "entity_type": entity_type,
            "integration_platform_version": "1.0"
        }
        
    def _calculate_overall_health(self, maintenance_results, prediction_results):
        """Calculate overall health status based on maintenance and prediction results"""
        # Default values
        maintenance_status = maintenance_results.get("status", "unknown")
        prediction_risk = prediction_results.get("risk_level", "unknown")
        
        health_scores = {
            "maintenance": {
                "healthy": 100,
                "info": 80,
                "warning": 50,
                "critical": 20,
                "unknown": 50
            },
            "prediction": {
                "low": 90,
                "medium": 50,
                "high": 20,
                "unknown": 50
            }
        }
        
        maintenance_score = health_scores["maintenance"].get(maintenance_status, 50)
        prediction_score = health_scores["prediction"].get(prediction_risk, 50)
        
        overall_score = (0.6 * maintenance_score) + (0.4 * prediction_score)
        
        if overall_score >= 80:
            status = "healthy"
        elif overall_score >= 50:
            status = "stable"
        elif overall_score >= 30:
            status = "degraded"
        else:
            status = "critical"
            
        return {
            "status": status,
            "score": round(overall_score, 1),
            "maintenance_contribution": maintenance_score,
            "prediction_contribution": prediction_score
        }
        
    def get_metrics(self):
        return {
            "responses_aggregated": self.responses_aggregated,
            "last_aggregation_time": self.last_aggregation_time,
            "status": "healthy"
        }
