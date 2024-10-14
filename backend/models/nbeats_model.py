"""
N-BEATS Model Implementation for Time Series Forecasting

This module implements the N-BEATS (Neural Basis Expansion Analysis for Interpretable Time Series Forecasting) model
using the Darts library. N-BEATS is a deep neural architecture based on backward and forward residual links and a very
deep stack of fully-connected layers.

The model is designed for univariate time series forecasting and provides interpretable forecasts without the need
for extensive feature engineering or external regressors.

Key Features:
- Implements the NBEATSModel from Darts
- Provides methods for training, prediction, historical forecasts, and backtesting
- Includes evaluation metrics for model performance assessment

Reference:
Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2020). 
N-BEATS: Neural basis expansion analysis for interpretable time series forecasting. 
International Conference on Learning Representations (ICLR).
https://arxiv.org/abs/1905.10437

Usage:
The NBEATSPredictor class encapsulates the N-BEATS model and provides an interface for training,
prediction, and evaluation. It can be used for both single-step and multi-step forecasting.

Note: This implementation uses PyTorch Lightning for training acceleration and Streamlit for progress visualization.
"""

import traceback
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import streamlit as st
import torch
from darts import TimeSeries
from darts.metrics import mape, mse, rmse
from darts.models import NBEATSModel

from backend.utils.scaling import scale_data, inverse_scale
from backend.utils.tensor_utils import ensure_tensor_float32


def determine_accelerator() -> str:
    """
    Determine the available accelerator for PyTorch.

    Returns:
        str: The available accelerator ('gpu', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        return "gpu"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class PrintEpochResults(pl.Callback):
    """
    Callback to print epoch results during model training.

    Args:
        progress_bar: Streamlit progress bar object.
        status_text: Streamlit text object for status updates.
        total_epochs (int): Total number of epochs for training.
    """

    def __init__(self, progress_bar, status_text, total_epochs: int):
        super().__init__()
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.total_epochs = total_epochs

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Called at the end of each training epoch.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning trainer.
            pl_module (pl.LightningModule): The PyTorch Lightning module.
        """
        current_epoch = trainer.current_epoch
        loss = trainer.callback_metrics["train_loss"].item()
        progress = (current_epoch + 1) / self.total_epochs
        self.progress_bar.progress(progress)
        self.status_text.text(
            f"Training N-BEATS model: Epoch {current_epoch + 1}/{self.total_epochs}, Loss: {loss:.4f}"
        )


class NBEATSPredictor:
    """
    N-BEATS model predictor for time series forecasting.

    Args:
        input_chunk_length (int): Length of input sequences.
        output_chunk_length (int): Length of output sequences (forecast horizon).
    """

    def __init__(self, input_chunk_length: int = 24, output_chunk_length: int = 12):
        self.n_epochs = 100
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length

        # Create progress bar and status text
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Create callbacks
        print_epoch_results = PrintEpochResults(progress_bar, status_text, self.n_epochs)

        self.model = NBEATSModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            generic_architecture=True,
            num_stacks=10,
            num_blocks=1,
            num_layers=4,
            layer_widths=512,
            expansion_coefficient_dim=5,
            trend_polynomial_degree=2,
            batch_size=800,
            n_epochs=self.n_epochs,
            nr_epochs_val_period=1,
            dropout=0.1,
            model_name="nbeats_interpretable_run",
            pl_trainer_kwargs={
                "accelerator": determine_accelerator(),
                "precision": "32-true",
                "enable_model_summary": True,
                "callbacks": [print_epoch_results],
                "log_every_n_steps": 1,
                "enable_progress_bar": False,
            },
        )
        self.scaler = None

    def train(self, data: TimeSeries) -> None:
        """
        Train the N-BEATS model on the given time series data.

        Args:
            data (TimeSeries): The input time series data for training.
        """
        print("Starting N-BEATS model training...")
        data_float32 = data.astype(np.float32)
        scaled_data, self.scaler = scale_data(data_float32)

        self.model.fit(scaled_data, verbose=True)
        print("N-BEATS model training completed successfully")

    def predict(self, horizon: int, series: Optional[TimeSeries] = None) -> TimeSeries:
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
        
        if series is None:
            series = self.model.training_series
        
        series_float32 = series.astype(np.float32)
        if self.scaler is None:
            scaled_series, self.scaler = scale_data(series_float32)
        else:
            scaled_series = self.scaler.transform(series_float32)
        
        forecast = self.model.predict(n=horizon, series=scaled_series)
        
        return inverse_scale(forecast, self.scaler)

    def historical_forecasts(
        self,
        series: TimeSeries,
        start: Union[pd.Timestamp, int, float],
        forecast_horizon: int,
        stride: int = 1,
        retrain: bool = False,
        verbose: bool = False,
    ) -> Optional[TimeSeries]:
        """
        Generate historical forecasts using the trained N-BEATS model.

        Args:
            series (TimeSeries): The full time series data.
            start (Union[pd.Timestamp, int, float]): The start point for historical forecasts.
            forecast_horizon (int): Number of time steps to forecast for each historical point.
            stride (int): The stride between forecast points.
            retrain (bool): Whether to retrain the model for each forecast point.
            verbose (bool): Whether to print verbose output.

        Returns:
            Optional[TimeSeries]: Historical forecasts as a TimeSeries object, or None if an error occurs.

        Raises:
            ValueError: If the model has not been trained or if the start parameter is invalid.
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
        print(f"Historical forecast requested from {start} for {forecast_horizon} steps")
        print(f"Series range: {series.start_time()} to {series.end_time()}")

        # Convert start to pd.Timestamp if it's not already
        if isinstance(start, int):
            start = series.time_index[start]
        elif isinstance(start, float):
            start_index = int(len(series) * start)
            start = series.time_index[start_index]

        if not isinstance(start, pd.Timestamp):
            raise ValueError("start must be a pd.Timestamp, int index, or float proportion")

        # Ensure start is within the series timeframe
        if start >= series.end_time():
            raise ValueError(f"Start time {start} is at or after the last timestamp {series.end_time()} of the series.")

        # Adjust forecast horizon if it goes beyond the end of the series
        if start + pd.Timedelta(days=forecast_horizon) > series.end_time():
            forecast_horizon = (series.end_time() - start).days
            print(f"Adjusted forecast horizon to {forecast_horizon} to fit within available data")

        scaled_series, _ = scale_data(series.astype(np.float32))

        try:
            historical_forecast = self.model.historical_forecasts(
                scaled_series,
                start=start,
                forecast_horizon=forecast_horizon,
                stride=stride,
                retrain=retrain,
                verbose=verbose,
                overlap_end=True,
            )
            print(f"Historical forecast generated successfully. Length: {len(historical_forecast)}")
            return self.scaler.inverse_transform(historical_forecast)
        except Exception as e:
            print(f"Error in historical forecasts:{e}")
            print(traceback.format_exc())
            return None

    def backtest(self, data: TimeSeries, forecast_horizon: int, start: Union[float, int]) -> Tuple[TimeSeries, Dict[str, float]]:
        print(f"Starting N-BEATS backtest. Data length: {len(data)}, Forecast horizon: {forecast_horizon}, Start: {start}")
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")

        # Convert start to pd.Timestamp
        if isinstance(start, int):
            start_timestamp = data.time_index[start]
        elif isinstance(start, float):
            start_index = int(len(data) * start)
            start_timestamp = data.time_index[start_index]
        else:
            raise ValueError("start must be a float between 0 and 1 or an integer index.")

        print(f"Backtest start timestamp: {start_timestamp}")

        data_float32 = data.astype(np.float32)
        if self.scaler is None:
            scaled_data, self.scaler = scale_data(data_float32)
        else:
            scaled_data = self.scaler.transform(data_float32)

        try:
            historical_forecasts = []
            for i in range(start, len(scaled_data) - forecast_horizon + 1):
                forecast = self.model.predict(n=forecast_horizon, series=scaled_data[:i])
                historical_forecasts.append(forecast)

            # Combine all forecasts into a single TimeSeries
            historical_forecasts = TimeSeries.from_series(pd.concat([f.pd_series() for f in historical_forecasts]))
            print(f"Historical forecasts generated successfully. Length: {len(historical_forecasts)}")
            
            historical_forecasts = inverse_scale(historical_forecasts, self.scaler)
            
            # Ensure the historical forecasts match the test data length
            actual_data = data.slice(start_timestamp, data.end_time())
            if len(historical_forecasts) != len(actual_data):
                print(f"Adjusting historical forecasts length from {len(historical_forecasts)} to {len(actual_data)}")
                historical_forecasts = historical_forecasts.slice(actual_data.start_time(), actual_data.end_time())
            
        except Exception as e:
            print(f"Error generating historical forecasts: {str(e)}")
            raise

        print(f"Actual data prepared. Length: {len(actual_data)}")

        metrics = self.evaluate(actual_data, historical_forecasts)
        print(f"Metrics calculated: {metrics}")

        return historical_forecasts, metrics

    def evaluate(
            self,
            actual: TimeSeries,
            predicted: TimeSeries) -> Dict[str, float]:
        """
        Evaluate the model's performance using various metrics.

        Args:
            actual (TimeSeries): The actual time series data.
            predicted (TimeSeries): The predicted time series data.

        Returns:
            Dict[str, float]: A dictionary containing evaluation metrics (MAPE, MSE, RMSE).
        """
        return {"MAPE": mape(actual, predicted), "MSE": mse(actual, predicted), "RMSE": rmse(actual, predicted)}

    def get_train_test_forecast(self, train_data: TimeSeries, test_data: TimeSeries) -> Dict[str, Union[TimeSeries, Dict[str, float]]]:
        """
        Generate a dictionary containing train data, test data, forecast, and metrics.

        Args:
            train_data (TimeSeries): The training data.
            test_data (TimeSeries): The test data.

        Returns:
            Dict[str, Union[TimeSeries, Dict[str, float]]]: A dictionary containing 'train', 'test', 'forecast', and 'metrics' keys.
        """
        # Combine train and test data
        full_data = train_data.append(test_data)

        # Generate forecast for the test period
        forecast_horizon = len(test_data)
        start = len(train_data) - forecast_horizon  # Start backtest from forecast_horizon steps before the end of train_data
        forecast, metrics = self.backtest(full_data, forecast_horizon, start)

        # Ensure forecast matches test_data length
        if len(forecast) != len(test_data):
            print(f"Adjusting forecast length from {len(forecast)} to {len(test_data)}")
            forecast = forecast.slice(test_data.start_time(), test_data.end_time())

        return {
            'train': train_data,
            'test': test_data,
            'forecast': forecast,
            'metrics': metrics
        }
