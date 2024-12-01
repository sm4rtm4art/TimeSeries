from typing import Dict, Union
import pandas as pd
from darts import TimeSeries
import logging

from .interfaces.base_model import TimeSeriesPredictor
from backend.utils.time_utils import TimeSeriesUtils

logger = logging.getLogger(__name__)

class ModelForecaster:
    @staticmethod
    def generate_forecasts(
        models: Dict[str, TimeSeriesPredictor],
        horizon: int,
        data: TimeSeries
    ) -> Dict[str, Dict[str, TimeSeries]]:
        """Generate forecasts for all trained models."""
        forecasts = {}
        
        for model_name, model in models.items():
            try:
                logger.info(f"Generating forecast for {model_name}")
                future_forecast = model.predict(horizon)
                
                # Generate future dates
                future_dates = pd.date_range(
                    start=data.end_time() + TimeSeriesUtils.get_timedelta(data, 1),
                    periods=horizon,
                    freq=data.freq_str
                )
                
                # Ensure forecast length matches dates
                if len(future_forecast) == len(future_dates):
                    future_forecast = TimeSeries.from_times_and_values(
                        future_dates,
                        future_forecast.values()
                    )
                    
                    forecasts[model_name] = {
                        'future': future_forecast
                    }
                    logger.info(f"Successfully generated forecast for {model_name}")
                else:
                    logger.warning(f"Forecast length mismatch for {model_name}")
                    
            except Exception as e:
                logger.error(f"Error generating forecast for {model_name}: {str(e)}")
                continue
                
        return forecasts 