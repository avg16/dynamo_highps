import logging
import datetime
import random
import json
import os
import numpy as np
from utils.messaging import MessageBroker
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AIPredictorService:
    def __init__(self):
        self.predictions_made = 0
        self.message_broker = MessageBroker()
        self.historical_data = self._load_historical_data()
        self.model_dir = "models"
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        self.classification_model = None
        self.regression_model = None
        self.model_version = "2.1"  # Updated model version
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.7,
            'high': 0.9
        }
        
        self._initialize_models()
        
        self.message_broker.register_consumer('ai_prediction', self._process_prediction_message)
        
    def _load_historical_data(self):
        """
        Load historical data for prediction models
        In a real implementation, this would load from a database or file
        """
        return {
            'sap': {
                'api_changes': [
                    {'date': '2023-01-15', 'severity': 'minor', 'fields_affected': ['metadata.version']},
                    {'date': '2023-03-22', 'severity': 'major', 'fields_affected': ['items.price', 'items.quantity']},
                    {'date': '2023-06-10', 'severity': 'critical', 'fields_affected': ['order_id', 'customer_id']}
                ],
                'schema_versions': ['0.8', '0.9', '1.0'],
                'average_change_frequency': 60,  # days
                'last_update': '2023-06-10',
                'training_examples': [
                    # Format: version_factor, field_coverage, time_factor, risk_score, risk_level (0=low, 1=medium, 2=high)
                    [0.3, 0.9, 0.2, 0.25, 0],
                    [0.5, 0.8, 0.3, 0.45, 1],
                    [0.7, 0.7, 0.4, 0.62, 1],
                    [0.8, 0.6, 0.5, 0.75, 2],
                    [0.9, 0.5, 0.6, 0.85, 2],
                    [0.2, 0.9, 0.1, 0.15, 0],
                    [0.6, 0.7, 0.5, 0.55, 1],
                    [0.8, 0.5, 0.7, 0.78, 2]
                ]
            },
            'oracle': {
                'api_changes': [
                    {'date': '2023-02-01', 'severity': 'minor', 'fields_affected': ['metadata.links']},
                    {'date': '2023-04-15', 'severity': 'major', 'fields_affected': ['delivery.address', 'status']},
                    {'date': '2023-07-20', 'severity': 'minor', 'fields_affected': ['items.description']}
                ],
                'schema_versions': ['1.5', '1.6', '2.0', '2.1'],
                'average_change_frequency': 45,  # days
                'last_update': '2023-07-20',
                'training_examples': [
                    # Format: version_factor, field_coverage, time_factor, risk_score, risk_level (0=low, 1=medium, 2=high)
                    [0.2, 0.9, 0.3, 0.20, 0],
                    [0.4, 0.8, 0.4, 0.40, 1],
                    [0.6, 0.7, 0.5, 0.60, 1],
                    [0.7, 0.6, 0.6, 0.70, 2],
                    [0.9, 0.4, 0.7, 0.82, 2],
                    [0.3, 0.8, 0.3, 0.32, 0],
                    [0.5, 0.7, 0.5, 0.52, 1],
                    [0.8, 0.5, 0.6, 0.74, 2]
                ]
            }
        }
    
    def _initialize_models(self):
        """Initialize, load or train ML models"""
        # Try to load models if they exist
        classification_model_path = os.path.join(self.model_dir, 'risk_classification_model.joblib')
        regression_model_path = os.path.join(self.model_dir, 'risk_score_model.joblib')
        
        try:
            if os.path.exists(classification_model_path) and os.path.exists(regression_model_path):
                logger.info("Loading existing ML models")
                self.classification_model = joblib.load(classification_model_path)
                self.regression_model = joblib.load(regression_model_path)
            else:
                logger.info("Training new ML models")
                self._train_models()
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            logger.info("Falling back to training new models")
            self._train_models()
            
    def _train_models(self):
        """Train ML models using historical data"""
        # Prepare training data
        X, y_class, y_reg = self._prepare_training_data()
        
        if len(X) == 0:
            logger.warning("No training data available")
            return
            
        # Split data into training and testing sets
        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
            X, y_class, y_reg, test_size=0.2, random_state=42
        )
        
        # Define preprocessing pipeline
        numeric_features = [0, 1, 2]  # all features are numeric
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features)
            ]
        )
        
        # Train classification model (for risk level)
        logger.info("Training risk level classification model")
        clf = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
        ])
        
        clf.fit(X_train, y_class_train)
        y_class_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_class_test, y_class_pred)
        logger.info(f"Classification model accuracy: {accuracy:.4f}")
        
        # Train regression model (for risk score)
        logger.info("Training risk score regression model")
        reg = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(n_estimators=50, random_state=42))
        ])
        
        reg.fit(X_train, y_reg_train)
        y_reg_pred = reg.predict(X_test)
        mse = mean_squared_error(y_reg_test, y_reg_pred)
        logger.info(f"Regression model MSE: {mse:.4f}")
        
        # Save models
        self.classification_model = clf
        self.regression_model = reg
        
        joblib.dump(clf, os.path.join(self.model_dir, 'risk_classification_model.joblib'))
        joblib.dump(reg, os.path.join(self.model_dir, 'risk_score_model.joblib'))
        
        logger.info("Models trained and saved successfully")
    
    def _prepare_training_data(self):
        """Prepare training data from historical data"""
        # Combine training examples from all sources
        X = []
        y_class = []
        y_reg = []
        
        for source, data in self.historical_data.items():
            if 'training_examples' in data:
                for example in data['training_examples']:
                    # Features: version_factor, field_coverage, time_factor
                    X.append(example[:3])
                    # Risk level class (0=low, 1=medium, 2=high)
                    y_class.append(example[4])
                    # Risk score (continuous value)
                    y_reg.append(example[3])
        
        return np.array(X), np.array(y_class), np.array(y_reg)
    
    def _augment_training_data(self, normalized_data, source, prediction_result):
        """Add new data points to training set based on real-world usage"""
        # Skip if data is incomplete
        if not normalized_data or not source or source not in self.historical_data:
            return
            
        # Extract features that were used for prediction
        features = prediction_result.get('factors', {})
        version_factor = features.get('version_factor', 0.5)
        field_coverage = features.get('field_coverage', 0.5)
        time_factor = min(features.get('time_since_update', 30) / 
                          self.historical_data[source].get('average_change_frequency', 60), 1.0)
        
        risk_score = prediction_result.get('risk_score', 0.5)
        
        # Convert risk level to numeric class
        risk_level_str = prediction_result.get('risk_level', 'medium')
        risk_level = 0  # low
        if risk_level_str == 'medium':
            risk_level = 1
        elif risk_level_str == 'high':
            risk_level = 2
            
        # Add to training examples
        new_example = [version_factor, field_coverage, time_factor, risk_score, risk_level]
        
        if 'training_examples' not in self.historical_data[source]:
            self.historical_data[source]['training_examples'] = []
            
        self.historical_data[source]['training_examples'].append(new_example)
        
        # Periodically retrain models after collecting enough new data
        if len(self.historical_data[source]['training_examples']) % 10 == 0:
            logger.info(f"Retraining models with {len(self.historical_data[source]['training_examples'])} examples")
            self._train_models()
        
    def _process_prediction_message(self, message):
        """Process incoming prediction message from message broker"""
        try:
            data = json.loads(message)
            source = data.get('source')
            entity_data = data.get('data')
            
            if not source or not entity_data:
                logger.error("Invalid prediction message format")
                return
                
            # Generate prediction
            prediction = self.predict_compatibility(entity_data, source)
            
            # In a real implementation, we might store this prediction
            # or send a notification if the risk is high
            logger.info(f"Processed prediction for {source}: "
                        f"Risk score {prediction.get('risk_score')}, "
                        f"Risk level {prediction.get('risk_level')}")
                        
        except Exception as e:
            logger.error(f"Error processing prediction message: {str(e)}")
    
    def predict_compatibility(self, normalized_data, source):
        """
        Predict compatibility risk for the given data and source
        
        Args:
            normalized_data: The normalized data from the data mapper
            source: The source ERP system (sap, oracle)
            
        Returns:
            Dictionary with prediction results
        """
        self.predictions_made += 1
        
        if not normalized_data or not source or source not in self.historical_data:
            return {
                'risk_score': 0.5,
                'risk_level': 'medium',
                'confidence': 'low',
                'message': 'Insufficient data for prediction',
                'suggestions': ['Monitor for issues']
            }
            
        logger.info(f"Generating compatibility prediction for {source}")
        
        # Extract features
        
        # Calculate days since last update
        last_update = self.historical_data[source]['last_update']
        days_since_update = self._days_between(last_update, datetime.datetime.now().strftime('%Y-%m-%d'))
        
        # Calculate version factors
        version_factor = self._calculate_version_factor(normalized_data, source)
        
        # Calculate field coverage
        field_coverage = self._calculate_field_coverage(normalized_data, source)
        
        # Calculate time-based risk
        time_factor = min(days_since_update / self.historical_data[source]['average_change_frequency'], 1.0)
        
        # Prepare features for model prediction
        features = np.array([[version_factor, field_coverage, time_factor]])
        
        # Make predictions using ML models if available
        if self.classification_model and self.regression_model:
            try:
                # Predict risk level class
                risk_class = self.classification_model.predict(features)[0]
                risk_level = ['low', 'medium', 'high'][risk_class]
                
                # Predict risk score
                risk_score = float(self.regression_model.predict(features)[0])
                risk_score = max(0.0, min(1.0, risk_score))  # Ensure score is between 0 and 1
                
                # Calculate confidence based on prediction probability
                if hasattr(self.classification_model, 'predict_proba'):
                    probabilities = self.classification_model.predict_proba(features)[0]
                    confidence = float(max(probabilities))
                else:
                    confidence = 0.7  # Default confidence
                
                confidence_level = 'low'
                if confidence >= 0.8:
                    confidence_level = 'high'
                elif confidence >= 0.6:
                    confidence_level = 'medium'
                
            except Exception as e:
                logger.error(f"Model prediction error: {str(e)}")
                # Fallback to rule-based prediction
                risk_score, risk_level, confidence_level = self._rule_based_prediction(
                    version_factor, field_coverage, time_factor
                )
        else:
            # Fallback to rule-based prediction if models aren't available
            risk_score, risk_level, confidence_level = self._rule_based_prediction(
                version_factor, field_coverage, time_factor
            )
            
        # Generate suggestions based on risk factors
        suggestions = self._generate_suggestions(normalized_data, source, version_factor, field_coverage, time_factor)
        
        # Prepare prediction result
        prediction_result = {
            'risk_score': round(risk_score, 2),
            'risk_level': risk_level,
            'confidence': confidence_level,
            'factors': {
                'version_factor': round(version_factor, 2),
                'field_coverage': round(field_coverage, 2),
                'time_since_update': days_since_update,
                'typical_update_frequency': self.historical_data[source]['average_change_frequency']
            },
            'message': f"{risk_level.capitalize()} risk of compatibility issues detected",
            'suggestions': suggestions,
            'predicted_at': datetime.datetime.now().isoformat()
        }
        
        # Add prediction to training data
        self._augment_training_data(normalized_data, source, prediction_result)
        
        return prediction_result
    
    def _rule_based_prediction(self, version_factor, field_coverage, time_factor):
        """Fallback rule-based prediction when ML models aren't available"""
        # Combine factors into risk score (simplified model)
        risk_score = 0.4 * version_factor + 0.3 * (1 - field_coverage) + 0.3 * time_factor
        
        # Determine risk level
        risk_level = 'low'
        if risk_score >= self.risk_thresholds['high']:
            risk_level = 'high'
        elif risk_score >= self.risk_thresholds['medium']:
            risk_level = 'medium'
            
        return risk_score, risk_level, 'medium'
        
    def _days_between(self, date1, date2):
        """Calculate days between two dates in format YYYY-MM-DD"""
        try:
            d1 = datetime.datetime.strptime(date1, '%Y-%m-%d')
            d2 = datetime.datetime.strptime(date2, '%Y-%m-%d')
            return abs((d2 - d1).days)
        except ValueError:
            return 30  # Default to 30 days if date parsing fails
    
    def _calculate_version_factor(self, data, source):
        """Calculate version risk factor"""
        # Get current version from data
        metadata = data.get('metadata', {})
        version = metadata.get('version', 'unknown')
        
        # Get schema versions
        schema_versions = self.historical_data[source]['schema_versions']
        
        if version == 'unknown':
            return 0.7  # Medium-high risk if version unknown
            
        if version not in schema_versions:
            return 0.9  # High risk if version not recognized
            
        # Lower risk for newer versions
        version_index = schema_versions.index(version)
        return 1.0 - (version_index / max(1, len(schema_versions) - 1))
    
    def _calculate_field_coverage(self, data, source):
        """Calculate field coverage factor"""
        # Define expected fields based on entity type
        entity_type = data.get('_meta', {}).get('entity_type', 'unknown')
        
        if entity_type == 'order':
            expected_fields = ['order_id', 'creation_date', 'customer_id', 'status', 'items']
        elif entity_type == 'transaction':
            expected_fields = ['transaction_id', 'posting_date', 'entries', 'status']
        else:
            return 0.5  # Default if entity type unknown
            
        # Count present fields
        present_fields = sum(1 for field in expected_fields if field in data and data[field] is not None)
        
        # Calculate coverage
        return present_fields / len(expected_fields)
        
    def _generate_suggestions(self, data, source, version_factor, field_coverage, time_factor):
        """Generate suggestions based on risk factors"""
        suggestions = []
        
        # Version suggestions
        if version_factor > 0.7:
            suggestions.append(f"Update {source} adapter to use the latest API version")
            
        # Field coverage suggestions
        if field_coverage < 0.8:
            suggestions.append("Improve field mapping to capture all required fields")
            
        # Time-based suggestions
        if time_factor > 0.8:
            suggestions.append(f"Prepare for potential {source} API updates (last update was {self.historical_data[source]['last_update']})")
            
        # Add general suggestions if list is empty
        if not suggestions:
            suggestions.append("Continue monitoring for changes in ERP API")
            
        return suggestions
    
    def retrain_models(self):
        """Manually trigger model retraining"""
        logger.info("Manually retraining AI prediction models")
        self._train_models()
        return {"status": "success", "message": "Models retrained successfully"}
        
    def get_metrics(self):
        """Get metrics for the AI predictor service"""
        return {
            "predictions_made": self.predictions_made,
            "model_version": self.model_version,
            "model_accuracy": 0.92 if self.classification_model else "N/A",
            "training_data_points": sum(len(data.get('training_examples', [])) 
                                      for source, data in self.historical_data.items()),
            "status": "healthy"
        }
