import logging
import traceback
from typing import Any, Dict, Optional

from darts import TimeSeries

from backend.models.data.nbeats_model import make_nbeats_forecast
from backend.models.data.prophet_model import make_prophet_forecast

logger = logging.getLogger(__name__)

def generate_forecast_for_model(
    model_name: str,
    trained_models: Dict[str, Any],
    data: TimeSeries,
    forecast_horizon: int
) -> Optional[TimeSeries]:
    """
    Generate forecast for a specific model.
    
    Args:
        model_name: Name of the model to use
        trained_models: Dictionary of trained models
        data: Input TimeSeries data
        forecast_horizon: Number of steps to forecast
    
    Returns:
        Optional[TimeSeries]: Forecasted values or None if forecast fails
    """
    forecast_functions = {
        "N-BEATS": make_nbeats_forecast,
        "Prophet": make_prophet_forecast,
    }
    
    try:
        if model_name not in forecast_functions:
            logger.warning(f"No forecast function found for {model_name}")
            return None
            
        forecast = forecast_functions[model_name](trained_models, data, forecast_horizon)
        if not isinstance(forecast, TimeSeries):
            logger.error(f"Forecast for {model_name} is not a TimeSeries object")
            return None
            
        return forecast
        
    except Exception as e:
        logger.error(f"Error generating forecast for {model_name}: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def generate_forecasts(
    trained_models: Dict[str, Any],
    data: TimeSeries,
    forecast_horizon: int
) -> Dict[str, TimeSeries]:
    """Generate forecasts for all trained models."""
    forecasts = {}
    
    for model_name in trained_models:
        forecast = generate_forecast_for_model(model_name, trained_models, data, forecast_horizon)
        if forecast is not None:
            forecasts[model_name] = forecast
    
    return forecasts

