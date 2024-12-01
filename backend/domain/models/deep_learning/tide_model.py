"""
TiDE (Time-series Dense Encoder) Model Implementation for Time Series Forecasting

This module implements the TiDE model for time series forecasting using the Darts library.
TiDE is a novel deep learning architecture designed specifically for time series forecasting,
combining the strengths of autoregressive models and neural networks.

TiDE introduces a dense encoder that captures complex temporal patterns and dependencies
in time series data. It's particularly effective for long-term forecasting and can handle
both univariate and multivariate time series.

Key Features:
- Implements the TiDEModel from Darts
- Provides methods for training, prediction, historical forecasts, and backtesting
- Includes evaluation metrics for model performance assessment
- Supports GPU acceleration for faster training and inference
- Implements data scaling for improved model performance

Reference:
Zhu, Q., & Laptev, N. (2022).
TiDE: Time-series Dense Encoder for Forecasting.
arXiv preprint arXiv:2304.08424.
https://arxiv.org/abs/2304.08
"""

import traceback
from typing import Dict, Tuple, Union, List, Callable, Any
import logging

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import streamlit as st
import torch
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, mse, rmse
from darts.models import TiDEModel
from pytorch_lightning.callbacks import EarlyStopping
from backend.utils.scaling import scale_data, inverse_scale
from backend.core.interfaces.base_model import DartsModelPredictor

# Configure logger
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
        self.status_text.text(f"Training TiDE model: Epoch {current_epoch + 1}/{self.total_epochs}, Loss: {loss:.4f}")

class TiDEPredictor(DartsModelPredictor):
    def __init__(self):
        self.hidden_size = 64
        self.dropout = 0.1
        self.n_epochs = 100
        self.batch_size = 32
        self.optimizer_kwargs = {"lr": 1e-3}
        super().__init__()  # Call parent init after setting model parameters

    def _create_model(self) -> Any:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        trainer_kwargs = {
            "accelerator": determine_accelerator(),
            "callbacks": [
                EarlyStopping(
                    monitor="train_loss",
                    patience=10,
                    min_delta=0.000001,
                    mode="min"
                ),
                PrintEpochResults(
                    progress_bar,
                    status_text,
                    self.n_epochs
                )
            ],
            "enable_progress_bar": True,
            "enable_model_summary": True,
            "log_every_n_steps": 1
        }
        
        return TiDEModel(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            optimizer_kwargs=self.optimizer_kwargs,
            pl_trainer_kwargs=trainer_kwargs
        )

    def _generate_forecast(self, horizon: int, scaled_data: TimeSeries) -> TimeSeries:
        return self.model.predict(n=horizon, series=scaled_data)

    def _generate_historical_forecasts(
        self,
        series: TimeSeries,  # Changed from scaled_data
        start: Union[pd.Timestamp, float],
        forecast_horizon: int,
        stride: int,
        retrain: bool = False,
        verbose: bool = False,
        **kwargs
    ) -> TimeSeries:
        return self.model.historical_forecasts(
            series=series,
            start=start,
            forecast_horizon=forecast_horizon,
            stride=stride,
            retrain=retrain,
            verbose=verbose,
            show_warnings=False
        )

def train_tide_model(data: TimeSeries) -> TiDEPredictor:
    model = TiDEPredictor()
    model.train(data)
    return model
