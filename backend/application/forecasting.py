"""
Forecasting Service for handling all forecasting-related operations
"""
import logging
from typing import Dict, Union, Any
import pandas as pd
from darts import TimeSeries
from darts.metrics import mape, rmse, mae

from backend.core.interfaces.base_model import BasePredictor
from backend.utils.time_utils import TimeSeriesUtils

logger = logging.getLogger(__name__)

class ForecastingService:
    @staticmethod
    def generate_forecasts(
        trained_models: Dict[str, BasePredictor],
        data: TimeSeries,
        forecast_horizon: int,
        backtests: Dict[str, Dict[str, Union[TimeSeries, Dict[str, float]]]]
    ) -> Dict[str, Dict[str, TimeSeries]]:
        """Generate forecasts for all trained models"""
        forecasts = {}
        
        if not isinstance(trained_models, dict):
            logger.error(f"Expected trained_models to be a dictionary, but got {type(trained_models)}")
            return forecasts

        for model_name, model in trained_models.items():
            try:
                future_forecast = model.predict(horizon=forecast_horizon)
                future_dates = pd.date_range(
                    start=data.end_time() + TimeSeriesUtils.get_timedelta(data, 1),
                    periods=forecast_horizon,
                    freq=data.freq_str
                )
                
                if len(future_forecast) == len(future_dates):
                    future_forecast = TimeSeries.from_times_and_values(
                        future_dates,
                        future_forecast.values()
                    )
                
                forecasts[model_name] = {
                    'future': future_forecast,
                    'backtest': backtests.get(model_name, {}).get('backtest'),
                    'metrics': backtests.get(model_name, {}).get('metrics')
                }
                
            except Exception as e:
                logger.error(f"Error generating forecast for {model_name}: {str(e)}")
                continue
                
        return forecasts

    @staticmethod
    def perform_backtesting(
        data: TimeSeries,
        trained_models: Dict[str, BasePredictor],
        horizon: int,
        stride: int = 1
    ) -> Dict[str, Dict[str, Any]]:
        """Perform backtesting for all models"""
        backtests = {}
        
        for model_name, model in trained_models.items():
            try:
                backtest_result = model.backtest(
                    data=data,
                    start=0.6,
                    forecast_horizon=horizon,
                    stride=stride
                )
                backtests[model_name] = backtest_result
                
            except Exception as e:
                logger.error(f"Error in backtesting {model_name}: {str(e)}")
                continue
                
        return backtests 