"""
Prophet Model Implementation for Time Series Forecasting

This module implements the Prophet model for time series forecasting using the Darts library.
Prophet is a procedure for forecasting time series data based on an additive model where non-linear
trends are fit with yearly, weekly, and daily seasonality, plus holiday effects.

Key Features:
- Implements the Prophet model from Darts
- Provides methods for training, prediction, historical forecasts, and backtesting
- Includes evaluation metrics for model performance assessment
"""

import logging
from typing import Dict, Union, Any
import pandas as pd
from darts import TimeSeries
from darts.models import Prophet
from darts.metrics import mae, mape, rmse, mse
from darts.dataprocessing.transformers import Scaler
from backend.core.interfaces.model import TimeSeriesPredictor

logger = logging.getLogger(__name__)

class ProphetModel(TimeSeriesPredictor):
    def __init__(self):
        super().__init__()
        self.model = Prophet()
        self.is_trained = False
        self.scaler = Scaler()
        self.train_data = None

    def train(self, data: TimeSeries) -> None:
        try:
            logger.info("Training Prophet model")
            self.train_data = data
            scaled_data = self.scaler.fit_transform(data)
            self.model.fit(scaled_data)
            self.is_trained = True
            logger.info("Prophet model trained successfully")
        except Exception as e:
            logger.error(f"Error training Prophet model: {str(e)}")
            raise

    def predict(self, horizon: int, data: TimeSeries = None) -> TimeSeries:
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before prediction")
            
            forecast = self.model.predict(n=horizon)
            forecast = self.scaler.inverse_transform(forecast)
            logger.info(f"Generated forecast of length {len(forecast)}")
            return forecast
            
        except Exception as e:
            logger.error(f"Error in Prophet prediction: {str(e)}")
            raise

    def backtest(
        self,
        data: TimeSeries,
        start: Union[float, str],
        forecast_horizon: int,
        stride: int = 1
    ) -> Dict[str, Any]:
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before backtesting")
                
            logger.info("Starting Prophet backtesting")
            
            scaled_data = self.scaler.transform(data)
            historical_forecasts = self.model.historical_forecasts(
                series=scaled_data,
                start=start,
                forecast_horizon=forecast_horizon,
                stride=stride,
                retrain=True
            )
            
            historical_forecasts = self.scaler.inverse_transform(historical_forecasts)
            actual_data = data[historical_forecasts.start_time():historical_forecasts.end_time()]
            
            metrics = {
                'MAPE': float(mape(actual_data, historical_forecasts)),
                'RMSE': float(rmse(actual_data, historical_forecasts)),
                'MSE': float(mse(actual_data, historical_forecasts))
            }
            
            return {
                'backtest': historical_forecasts,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Error in Prophet backtesting: {str(e)}")
            raise