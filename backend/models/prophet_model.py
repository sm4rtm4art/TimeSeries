"""
Prophet Model Implementation for Time Series Forecasting

This module implements the Prophet model for time series forecasting using the Darts library.
Prophet is a procedure for forecasting time series data based on an additive model where non-linear
trends are fit with yearly, weekly, and daily seasonality, plus holiday effects.

Prophet is designed to handle time series with strong seasonal effects and several seasons of historical data.
It is robust to missing data and shifts in the trend, and typically handles outliers well.

Key Features:
- Implements the Prophet model from Darts
- Provides methods for training, prediction, historical forecasts, and backtesting
- Includes evaluation metrics for model performance assessment

Reference:
Taylor, S. J., & Letham, B. (2018). 
Forecasting at scale. 
The American Statistician, 72(1), 37-45.
https://doi.org/10.1080/00031305.2017.1380080

Usage:
The ProphetModel class encapsulates the Prophet model and provides an interface for training,
prediction, and evaluation. It can be used for both single-step and multi-step forecasting.

Note: This implementation uses Streamlit for progress visualization during historical forecasts.
"""

import logging
from typing import Dict, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
import traceback

import streamlit as st
from darts import TimeSeries
from darts.metrics import mae, mape, rmse, mse
from darts.models import Prophet
from darts.dataprocessing.transformers import Scaler

from .base_model import BasePredictor
from backend.utils.scaling import scale_data, inverse_scale

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ProphetModel(BasePredictor):
    """
    A wrapper class for the Prophet model from the Darts library.

    Args:
        input_chunk_length (int): The length of input sequences (default: 24).
        output_chunk_length (int): The length of output sequences (default: 12).
    """

    def __init__(self):
        super().__init__(model_name="Prophet")

    def _create_model(self) -> Any:
        return Prophet()

    def _train_model(self, scaled_data: TimeSeries) -> None:
        self.model.fit(scaled_data)

    def _generate_forecast(self, horizon: int, scaled_data: TimeSeries = None) -> TimeSeries:
        """Generate forecast using Prophet model"""
        try:
            # Prophet doesn't need the series parameter for prediction
            forecast = self.model.predict(n=horizon)
            return forecast
        except Exception as e:
            logger.error(f"Error in Prophet forecast generation: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _generate_historical_forecasts(
        self,
        series: TimeSeries,
        start: Union[pd.Timestamp, float],
        forecast_horizon: int,
        stride: int = 1,
        **kwargs
    ) -> TimeSeries:
        """Generate historical forecasts using Prophet model"""
        return self.model.historical_forecasts(
            series=series,
            start=start,
            forecast_horizon=forecast_horizon,
            stride=stride,
            retrain=True,
            verbose=False
        )

    def backtest(
        self,
        data: TimeSeries,
        start: Union[pd.Timestamp, float],
        forecast_horizon: int,
        stride: int = 1
    ) -> Dict[str, Union[TimeSeries, Dict[str, float]]]:
        """Generate historical forecasts and calculate metrics."""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before backtesting")
                
            logger.info("Starting Prophet backtesting")
            
            # Generate historical forecasts
            historical_forecasts = self.historical_forecasts(
                series=data,
                start=start,
                forecast_horizon=forecast_horizon,
                stride=stride
            )
            
            # Calculate metrics
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
            logger.error(traceback.format_exc())
            raise

    def evaluate(self, actual: TimeSeries, predicted: TimeSeries) -> Dict[str, Optional[float]]:
        """
        Evaluate the model's performance using various metrics.

        Args:
            actual (TimeSeries): The actual time series data.
            predicted (TimeSeries): The predicted time series data.

        Returns:
            Dict[str, Optional[float]]: A dictionary containing evaluation metrics (MAPE, RMSE, MAE).
        """

        try:
            return {"MAPE": mape(actual, predicted), "RMSE": rmse(actual, predicted), "MAE": mae(actual, predicted)}
        except Exception as e:
            logger.error(f"Error during Prophet model evaluation: {str(e)}")
            return {"MsAPE": None, "RMSE": None, "MAE": None}


def train_prophet_model(data: TimeSeries) -> ProphetModel:
    """
    Train a Prophet model on the given time series data.

    Args:
        data (TimeSeries): The input time series data for training.

    Returns:
        ProphetModel: A trained ProphetModel instance.
    """
    model = ProphetModel()
    model.train(data)
    return model


def make_prophet_forecast(trained_models: Dict[str, 'ProphetModel'], data: TimeSeries, forecast_horizon: int) -> TimeSeries:
    """
    Generate forecasts using the Prophet model.
    
    Args:
        trained_models: Dictionary of trained models
        data: Input TimeSeries data
        forecast_horizon: Number of steps to forecast
    
    Returns:
        TimeSeries: Forecasted values
    """
    try:
        logger.info(f"Generating Prophet forecast for horizon: {forecast_horizon}")
        
        if "Prophet" not in trained_models:
            raise KeyError("Prophet model not found in trained models")
            
        model = trained_models["Prophet"]
        if not model.is_trained:
            raise ValueError("Prophet model is not trained")
        
        # Generate forecast
        forecast = model.predict(horizon=forecast_horizon, data=data)
        
        logger.info(f"Prophet forecast generated successfully. Length: {len(forecast)}")
        return forecast
        
    except Exception as e:
        logger.error(f"Error in Prophet forecast: {str(e)}")
        logger.error(traceback.format_exc())
        raise
