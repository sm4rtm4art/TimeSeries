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
from .base_model import BasePredictor

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

class TiDEPredictor(BasePredictor):
    """TiDE (Time-series Dense Encoder) model predictor class."""
    
    def __init__(self):
        super().__init__(model_name="TiDE")
        self.input_chunk_length = 24
        self.output_chunk_length = 12
        self.hidden_size = 64
        self.dropout = 0.1
        self.n_epochs = 100
        self.batch_size = 32
        self.optimizer_kwargs = {"lr": 1e-3}

    def _create_model(self) -> Any:
        callbacks = [
            PrintEpochResults(st.progress(0), st.empty(), self.n_epochs),
            EarlyStopping(
                monitor="train_loss",
                patience=5,
                min_delta=0.001,
                mode="min"
            )
        ]
        
        return TiDEModel(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            batch_size=self.batch_size,
            optimizer_kwargs=self.optimizer_kwargs,
            pl_trainer_kwargs={
                "accelerator": determine_accelerator(),
                "max_epochs": self.n_epochs,
                "callbacks": callbacks,
                "enable_progress_bar": False
            }
        )

    def _train_model(self, scaled_data: TimeSeries) -> None:
        self.model.fit(scaled_data, verbose=False)

    def _generate_forecast(self, horizon: int, scaled_data: TimeSeries) -> TimeSeries:
        return self.model.predict(n=horizon, series=scaled_data)

    def _generate_historical_forecasts(
        self,
        scaled_data: TimeSeries,
        start: Union[pd.Timestamp, float],
        forecast_horizon: int,
        stride: int,
        retrain: bool = False,
        verbose: bool = False,
        **kwargs
    ) -> TimeSeries:
        return self.model.historical_forecasts(
            series=scaled_data,
            start=start,
            forecast_horizon=forecast_horizon,
            stride=stride,
            retrain=retrain,
            verbose=verbose,
            show_warnings=False
        )

    def backtest(
        self,
        data: TimeSeries,
        start: Union[pd.Timestamp, float],
        forecast_horizon: int,
        stride: int = 1,
    ) -> Dict[str, Union[TimeSeries, Dict[str, float]]]:
        """Perform backtesting of the model.
        
        Args:
            data: TimeSeries data for backtesting
            start: Start point for backtesting
            forecast_horizon: Number of steps to forecast
            stride: Stride for moving the prediction window
            
        Returns:
            Dictionary containing backtest results and metrics
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before backtesting")

            logger.info("Starting TiDE backtest")
            
            # Generate historical forecasts
            historical_forecasts = self._generate_historical_forecasts(
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
            logger.error(f"Error in TiDE backtest: {str(e)}")
            logger.error(traceback.format_exc())
            raise

def train_tide_model(data: TimeSeries) -> TiDEPredictor:
    model = TiDEPredictor()
    model.train(data)
    return model
