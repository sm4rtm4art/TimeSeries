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
from typing import Dict, Optional

import streamlit as st
from darts import TimeSeries
from darts.metrics import mae, mape, rmse
from darts.models import Prophet

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ProphetModel:
    """
    A wrapper class for the Prophet model from the Darts library.

    Args:
        input_chunk_length (int): The length of input sequences (default: 24).
        output_chunk_length (int): The length of output sequences (default: 12).
    """

    def __init__(self, input_chunk_length: int = 24, output_chunk_length: int = 12):
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_mode="multiplicative",
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            holidays_prior_scale=10.0,
            changepoint_range=0.9,
        )
        self.data: Optional[TimeSeries] = None
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length

    def train(self, data: TimeSeries) -> None:
        """
        Train the Prophet model on the given time series data.

        Args:
            data (TimeSeries): The input time series data for training.

        Raises:
            Exception: If an error occurs during model training.
        """
        logger.info(f"Training Prophet model with data of length {len(data)}")
        self.data = data
        try:
            self.model.fit(data)
            logger.info("Prophet model training completed")
        except Exception as e:
            logger.error(f"Error during Prophet model training: {str(e)}")
            raise

    def predict(self, horizon: int) -> TimeSeries:
        """
        Generate predictions using the trained Prophet model.

        Args:
            horizon (int): Number of time steps to forecast.

        Returns:
            TimeSeries: Forecasted time series.

        Raises:
            ValueError: If the model has not been trained.
            Exception: If an error occurs during prediction.
        """
        if self.data is None:
            raise ValueError("Model has not been trained. Call train() before predict().")

        logger.info(f"Predicting with Prophet model. Horizon: {horizon}")

        try:
            forecast = self.model.predict(n=horizon, num_samples=1)
            # Ensure the forecast is deterministic
            forecast = forecast.mean()  # Take the mean of the probabilistic forecast
            logger.info(f"Generated forecast with length {len(forecast)}")
            return forecast
        except Exception as e:
            logger.error(f"Error during Prophet model prediction: {str(e)}")
            raise

    def historical_forecasts(self, series: TimeSeries, start: int, forecast_horizon: int, stride: int = 1, retrain: bool = True, verbose: bool = False) -> TimeSeries:
        """
        Generate historical forecasts using the Prophet model.

        Args:
            series (TimeSeries): The full time series data.
            start (int): The start index for historical forecasts.
            forecast_horizon (int): Number of time steps to forecast for each historical point.
            stride (int): The stride between forecast points (default: 1).
            retrain (bool): Whether to retrain the model for each forecast point (default: True).
            verbose (bool): Whether to print verbose output (default: False).

        Returns:
            TimeSeries: Historical forecasts as a TimeSeries object.

        Raises:
            ValueError: If the start parameter is not an integer.
            Exception: If an error occurs during historical forecasts generation.
        """
        logger.info(f"Generating historical forecasts. Start: {start}, Horizon: {forecast_horizon}, Stride: {stride}")

        try:
            if not isinstance(start, int):
                raise ValueError("Start must be an integer index")

            n_iterations = (len(series) - start - forecast_horizon) // stride + 1

            progress_bar = st.progress(0)
            status_text = st.empty()

            historical_forecasts = []
            for i in range(n_iterations):
                current_start = start + i * stride
                train_data = series.slice(series.start_time(), series.time_index[current_start])
                if retrain:
                    self.train(train_data)
                forecast = self.predict(forecast_horizon)
                historical_forecasts.append(forecast)

                progress = (i + 1) / n_iterations
                progress_bar.progress(progress)
                status_text.text(f"Prophet Historical Forecasts: {i+1}/{n_iterations}")

            # Combine all forecasts into a single TimeSeries
            combined_forecast = historical_forecasts[0]
            for forecast in historical_forecasts[1:]:
                combined_forecast = combined_forecast.append(forecast)

            logger.info(f"Generated historical forecasts with length {len(combined_forecast)}")
            return combined_forecast
        except Exception as e:
            logger.error(f"Error during historical forecasts generation: {str(e)}")
            raise

    def backtest(self, data: TimeSeries, forecast_horizon: int, start: int) -> TimeSeries:
        """
        Perform backtesting on the Prophet model.

        Args:
            data (TimeSeries): The full time series data for backtesting.
            forecast_horizon (int): Number of time steps to forecast for each backtest point.
            start (int): The start index for backtesting.

        Returns:
            TimeSeries: Backtesting forecast results.
        """
        backtest_forecast = self.historical_forecasts(data, start, forecast_horizon, retrain=True)
        return backtest_forecast

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
