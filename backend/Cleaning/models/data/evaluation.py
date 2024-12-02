from typing import Dict, Any
import logging
import numpy as np
from darts import TimeSeries
from backend.utils.metrics import calculate_metrics

logger = logging.getLogger(__name__)


def evaluate_models(trained_models: Dict[str, Any], test_data: TimeSeries) -> Dict[str, Dict[str, float]]:
    """
    Evaluate trained models against test data.
    
    Args:
        trained_models: Dictionary of trained model instances
        test_data: Test data to evaluate against
        
    Returns:
        Dict containing evaluation metrics for each model
    """
    metrics = {}
    
    try:
        # Convert test data to float32
        test_data = test_data.astype(np.float32)
        
        for model_name, model in trained_models.items():
            try:
                # Generate forecast for test period
                forecast_horizon = len(test_data)
                
                # For NBEATS model, handle scaling separately
                if hasattr(model, 'scaler') and model.scaler is not None:
                    # Scale test data for comparison
                    scaled_test = TimeSeries.from_times_and_values(
                        times=test_data.time_index,
                        values=model.scaler.transform(test_data.values().astype(np.float32))
                    )
                    forecast = model.predict(forecast_horizon)
                else:
                    # For other models, use test data directly
                    forecast = model.predict(forecast_horizon)
                
                # Calculate metrics
                model_metrics = calculate_metrics(test_data, forecast)
                metrics[model_name] = model_metrics
                
                logger.info(f"Evaluated {model_name} model. Metrics: {model_metrics}")
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name} model: {str(e)}")
                metrics[model_name] = {
                    "MAE": None,
                    "MSE": None,
                    "RMSE": None,
                    "MAPE": None,
                    "sMAPE": None
                }
            
    except Exception as e:
        logger.error(f"Error evaluating models: {str(e)}")
        raise
        
    return metrics
