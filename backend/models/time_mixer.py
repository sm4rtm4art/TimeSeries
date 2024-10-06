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

        print(f"Training TSMixer model with data of length {len(data)}")
        scaled_data = self.scaler.fit_transform(data)
        scaled_data_32 = scaled_data.astype(np.float32)

        # Create callbacks
        early_stopping = EarlyStopping(monitor="train_loss", patience=15, min_delta=0.000001, mode="min")
        print_epoch_results = PrintEpochResults(progress_bar, status_text, self.n_epochs)

        # Create the model
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

        # Train the model
        self.model.fit(scaled_data_32, verbose=True)
        print("TSMixer model training completed")
        st.text("TSMixer model training completed")

    def predict(self, horizon: int, data: TimeSeries = None) -> TimeSeries:
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() before predict().")

        print(f"Predicting with TSMixer model. Horizon: {horizon}")

        if data is not None:
            scaled_data = self.scaler.transform(data)
            scaled_data_32 = scaled_data.astype(np.float32)
        else:
            scaled_data_32 = self.scaler.transform(self.model.training_series)

        forecast = self.model.predict(n=horizon, series=scaled_data_32)
        unscaled_forecast = self.scaler.inverse_transform(forecast)

        print(f"Generated forecast with length {len(unscaled_forecast)}")
        return unscaled_forecast

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
        print(f"Historical forecast requested from {start} for {forecast_horizon} steps")
        print(f"Series range: {series.start_time()} to {series.end_time()}")

        scaled_series = self.scaler.transform(series.astype(np.float32))

        try:
            historical_forecast = self.model.historical_forecasts(
                scaled_series,
                start=start,
                forecast_horizon=forecast_horizon,
                stride=stride,
                retrain=retrain,
                verbose=verbose,
                last_points_only=False,
            )
            print(f"Historical forecast generated successfully. Length: {len(historical_forecast)}")
            return self.scaler.inverse_transform(historical_forecast)
        except Exception as e:
            print("Error in historical forecasts:")
            print(traceback.format_exc())
            return None

    def backtest(
        self, data: TimeSeries, forecast_horizon: int, start: Union[float, int]
    ) -> Tuple[TimeSeries, Dict[str, float]]:
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")

        # Convert start to a float between 0 and 1 if it's an integer
        if isinstance(start, int):
            start = start / len(data)

        if not 0 <= start < 1:
            raise ValueError(
                "start must be a float between 0 and 1 or an integer index less than the length of the data."
            )

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

        # Inverse transform the forecasts
        historical_forecasts = self.scaler.inverse_transform(historical_forecasts)

        # Calculate metrics
        metrics = calculate_metrics(backtest_series, historical_forecasts)

        return historical_forecasts, metrics


def train_tsmixer_model(
    data: TimeSeries, input_chunk_length: int, output_chunk_length: int, **kwargs
) -> TSMixerPredictor:
    model = TSMixerPredictor(input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length)
    model.train(data)
    return model
