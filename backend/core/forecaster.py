from typing import Dict, Union
from darts import TimeSeries
import logging
from .interfaces.base_model import BasePredictor

logger = logging.getLogger(__name__)

class ModelForecaster:
    @staticmethod
    def generate_forecast(
        model: BasePredictor,
        horizon: int,
        data: TimeSeries = None
    ) -> TimeSeries:
        """Generate forecast for a single model"""
        return model.predict(horizon=horizon, data=data)

    @staticmethod
    def generate_forecasts(
        models: Dict[str, BasePredictor],
        horizon: int,
        data: TimeSeries = None
    ) -> Dict[str, Dict[str, TimeSeries]]:
        """Generate forecasts for multiple models"""
        forecasts = {}
        
        for model_name, model in models.items():
            try:
                future_forecast = ModelForecaster.generate_forecast(model, horizon, data)
                forecasts[model_name] = {
                    'future': future_forecast
                }
            except Exception as e:
                logger.error(f"Error generating forecast for {model_name}: {str(e)}")
                continue
                
        return forecasts 