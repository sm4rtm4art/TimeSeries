"""
TSMixer Model Implementation for Time Series Forecasting

This module implements the TSMixer (Time Series Mixer) model for time series forecasting using the Darts library.
TSMixer is a novel architecture that combines the strengths of Transformer and MLP-Mixer models, specifically
designed for time series forecasting tasks.

TSMixer introduces a time-wise mixing mechanism that captures temporal dependencies effectively, while
maintaining computational efficiency. It's particularly well-suited for multivariate time series forecasting
and can handle both short-term and long-term dependencies.

Key Features:
- Implements the TSMixerModel from Darts
- Provides methods for training, prediction, historical forecasts, and backtesting
- Includes evaluation metrics for model performance assessment
- Supports GPU acceleration for faster training and inference

Reference:
Liu, H., Wu, J., Xie, T., Wen, R., & Huang, T. (2023).
TSMixer: An All-MLP Architecture for Time Series Forecasting.
arXiv preprint arXiv:2303.06053.
https://arxiv.org/abs/2303.06053

Usage:
The TSMixerPredictor class encapsulates the TSMixer model and provides an interface for training,
prediction, and evaluation. It can be used for both univariate and multivariate time series forecasting.

Note: This implementation uses PyTorch Lightning for training acceleration and Streamlit for progress visualization.
"""

from darts import TimeSeries
from darts.models import TSMixerModel
from darts.dataprocessing.transformers import Scaler
from typing import Union, Dict, Tuple
import torch.nn as nn
import numpy as np
import streamlit as st
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
import traceback
import pandas as pd
from backend.utils.metrics import calculate_metrics
import logging

logger = logging.getLogger(__name__)


def determine_accelerator():
    if torch.cuda.is_available():
        return "gpu"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class PrintEpochResults(pl.Callback):
    def __init__(self, progress_bar, status_text, total_epochs):
        super().__init__()
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.total_epochs = total_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        loss = trainer.callback_metrics["train_loss"].item()
        progress = (current_epoch + 1) / self.total_epochs
        self.progress_bar.progress(progress)
        self.status_text.text(
            f"Training TSMixer model: Epoch {current_epoch + 1}/{self.total_epochs}, Loss: {loss:.4f}"
        )


class TSMixerPredictor:
    def __init__(self, input_chunk_length=24, output_chunk_length=12):
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.n_epochs = 100
        self.model = None
        self.scaler = Scaler()

    def train(self, data: TimeSeries) -> None:
        st.text("Training TSMixer model...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            logger.info(f"Training TSMixer model with data of length {len(data)}")
            scaled_data = self.scaler.fit_transform(data.astype(np.float32))

            early_stopping = EarlyStopping(monitor="train_loss", patience=15, min_delta=0.000001, mode="min")
            print_epoch_results = PrintEpochResults(progress_bar, status_text, self.n_epochs)

            self.model = TSMixerModel(
                input_chunk_length=self.input_chunk_length,
                output_chunk_length=self.output_chunk_length,
                n_epochs=self.n_epochs,
                pl_trainer_kwargs={
                    "accelerator": determine_accelerator(),
                    "precision": "32-true",
                    "enable_model_summary": True,
                    "callbacks": [early_stopping, print_epoch_results],
                    "log_every_n_steps": 1,
                    "enable_progress_bar": False,
                },
            )

            self.model.fit(scaled_data, verbose=True)
            logger.info("TSMixer model training completed")
            st.text("TSMixer model training completed")
        except Exception as e:
            logger.error(f"Error during TSMixer model training: {str(e)}")
            st.error(f"Error during TSMixer model training: {str(e)}")
            raise

    def predict(self, horizon: int, data: TimeSeries = None) -> TimeSeries:
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() before predict().")

        try:
            logger.info(f"Predicting with TSMixer model. Horizon: {horizon}")

            if data is not None:
                scaled_data = self.scaler.transform(data.astype(np.float32))
            else:
                scaled_data = self.scaler.transform(self.model.training_series)

            forecast = self.model.predict(n=horizon, series=scaled_data)
            unscaled_forecast = self.scaler.inverse_transform(forecast)

            # Ensure the forecast has the correct length
            if len(unscaled_forecast) != horizon:
                logger.warning(f"Forecast length ({len(unscaled_forecast)}) doesn't match horizon ({horizon}). Adjusting...")
                if len(unscaled_forecast) > horizon:
                    unscaled_forecast = unscaled_forecast[:horizon]
                else:
                    pad_length = horizon - len(unscaled_forecast)
                    pad_values = np.full((pad_length, unscaled_forecast.width), np.nan)
                    pad_index = pd.date_range(start=unscaled_forecast.end_time() + unscaled_forecast.freq, periods=pad_length, freq=unscaled_forecast.freq)
                    pad_series = TimeSeries.from_times_and_values(pad_index, pad_values)
                    unscaled_forecast = unscaled_forecast.append(pad_series)

            logger.info(f"Generated forecast with length {len(unscaled_forecast)}")
            return unscaled_forecast
        except Exception as e:
            logger.error(f"Error during TSMixer prediction: {str(e)}")
            raise

    def historical_forecasts(
        self,
        series: TimeSeries,
        start: pd.Timestamp,
        forecast_horizon: int,
        stride: int = 1,
        retrain: bool = False,
        verbose: bool = False,
    ) -> TimeSeries:
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")

        try:
            logger.info(f"Historical forecast requested from {start} for {forecast_horizon} steps")
            logger.info(f"Series range: {series.start_time()} to {series.end_time()}")

            scaled_series = self.scaler.transform(series.astype(np.float32))

            historical_forecast = self.model.historical_forecasts(
                scaled_series,
                start=start,
                forecast_horizon=forecast_horizon,
                stride=stride,
                retrain=retrain,
                verbose=verbose,
                last_points_only=False,
            )

            unscaled_forecast = self.scaler.inverse_transform(historical_forecast)

            # Ensure the forecast has the correct length
            if len(unscaled_forecast) != forecast_horizon:
                logger.warning(f"Historical forecast length ({len(unscaled_forecast)}) doesn't match forecast_horizon ({forecast_horizon}). Adjusting...")
                if len(unscaled_forecast) > forecast_horizon:
                    unscaled_forecast = unscaled_forecast[:forecast_horizon]
                else:
                    pad_length = forecast_horizon - len(unscaled_forecast)
                    pad_values = np.full((pad_length, unscaled_forecast.width), np.nan)
                    pad_index = pd.date_range(start=unscaled_forecast.end_time() + unscaled_forecast.freq, periods=pad_length, freq=unscaled_forecast.freq)
                    pad_series = TimeSeries.from_times_and_values(pad_index, pad_values)
                    unscaled_forecast = unscaled_forecast.append(pad_series)

            logger.info(f"Historical forecast generated successfully. Length: {len(unscaled_forecast)}")
            return unscaled_forecast
        except Exception as e:
            logger.error(f"Error in historical forecasts: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def backtest(
        self, data: TimeSeries, forecast_horizon: int, start: Union[float, int]
    ) -> Tuple[TimeSeries, Dict[str, float]]:
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")

        try:
            # Convert start to a float between 0 and 1 if it's an integer
            if isinstance(start, int):
                start = start / len(data)

            if not 0 <= start < 1:
                raise ValueError("start must be a float between 0 and 1 or an integer index less than the length of the data.")

            scaled_data = self.scaler.transform(data.astype(np.float32))

            # Calculate the start index
            start_index = int(len(scaled_data) * start)

            # Perform backtesting
            backtest_series = scaled_data[start_index:]
            historical_forecasts = self.model.historical_forecasts(
                scaled_data,
                start=start_index,
                forecast_horizon=forecast_horizon,
                stride=1,
                retrain=False,
                verbose=True,
                last_points_only=False,
            )

            # Convert historical_forecasts to TimeSeries if it's a list
            if isinstance(historical_forecasts, list):
                logger.info(f"Converting historical_forecasts from list to TimeSeries. List length: {len(historical_forecasts)}")
                if len(historical_forecasts) == 1 and isinstance(historical_forecasts[0], TimeSeries):
                    historical_forecasts = historical_forecasts[0]
                else:
                    # Combine multiple forecasts if necessary
                    combined_values = np.concatenate([f.values() for f in historical_forecasts], axis=0)
                    historical_forecasts = TimeSeries.from_values(combined_values)

            # Inverse transform the forecasts
            historical_forecasts = self.scaler.inverse_transform(historical_forecasts)

            # Ensure the historical_forecasts has the correct time index
            actual_data = data[start_index:start_index + forecast_horizon]
            historical_forecasts = historical_forecasts.pd_dataframe()
            historical_forecasts.index = actual_data.time_index
            historical_forecasts = TimeSeries.from_dataframe(historical_forecasts)

            # Calculate metrics
            metrics = calculate_metrics(actual_data, historical_forecasts)

            logger.info(f"Backtest completed. Forecast length: {len(historical_forecasts)}")
            logger.info(f"Backtest metrics: {metrics}")

            return historical_forecasts, metrics
        except Exception as e:
            logger.error(f"Error during TSMixer backtesting: {str(e)}")
            logger.error(traceback.format_exc())
            return None, {}

    def train_tsmixer_model(
        self, data: TimeSeries, input_chunk_length: int, output_chunk_length: int, **kwargs
    ):
        model = TSMixerPredictor(input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length)
        model.train(data)
        return model

def train_tsmixer_model(data: TimeSeries):
    model = TSMixerPredictor()
    model.train(data)
    return model
