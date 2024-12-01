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
from typing import Dict, Union, Tuple, Any
import logging

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import streamlit as st
import torch
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, mse, rmse
from darts.models import NBEATSModel
from pytorch_lightning.callbacks import EarlyStopping

from ...core.interfaces.base_model import BasePredictor

# Configure logger
logger = logging.getLogger(__name__)


def determine_accelerator() -> str:
    """
    Determine the available accelerator for PyTorch.
    For Apple Silicon Macs, prefer MPS over CPU.

    Returns:
        str: The available accelerator ('gpu', 'mps', or 'cpu').
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "gpu"
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


class NBEATSPredictor(BasePredictor):
    def __init__(self):
        super().__init__(model_name="N-BEATS")
        self.input_chunk_length = 24
        self.output_chunk_length = 12
        self.n_epochs = 100
        self.training_config = {
            'accelerator': determine_accelerator(),
            'precision': '32-true',
            'force_dtype': torch.float32
        }

    def _create_model(self) -> Any:
        callbacks = [
            EarlyStopping(
                monitor="train_loss",
                patience=10,
                min_delta=0.000001,
                mode="min"
            ),
            PrintEpochResults(
                st.progress(0),
                st.empty(),
                self.n_epochs
            )
        ]

        return NBEATSModel(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            generic_architecture=True,
            num_stacks=10,
            num_blocks=1,
            num_layers=4,
            layer_widths=512,
            n_epochs=self.n_epochs,
            pl_trainer_kwargs={
                'accelerator': self.training_config['accelerator'],
                'callbacks': callbacks,
                'enable_progress_bar': True,
                'enable_model_summary': True,
                'log_every_n_steps': 1,
                'precision': self.training_config['precision'],
                'devices': 1
            },
            batch_size=32,
            model_name="nbeats",
            force_reset=True,
            save_checkpoints=True,
            optimizer_kwargs={'lr': 1e-3}
        )

    def _train_model(self, scaled_data: TimeSeries) -> None:
        self.model.fit(scaled_data, verbose=True)

    def _generate_forecast(self, horizon: int, scaled_data: TimeSeries) -> TimeSeries:
        return self.model.predict(n=horizon, series=scaled_data)

    def _generate_historical_forecasts(
        self,
        series: TimeSeries,
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
                raise ValueError(f"{self.model_name} must be trained before backtesting")
                
            logger.info(f"Starting {self.model_name} backtesting")
            
            # Generate historical forecasts
            historical_forecasts = self.historical_forecasts(
                series=data,
                start=start,
                forecast_horizon=forecast_horizon,
                stride=stride,
                retrain=False,
                verbose=True
            )
            
            # Calculate metrics
            actual_data = data[historical_forecasts.start_time():historical_forecasts.end_time()]
            metrics = {
                'MAPE': mape(actual_data, historical_forecasts),
                'MSE': mse(actual_data, historical_forecasts),
                'RMSE': rmse(actual_data, historical_forecasts)
            }
            
            logger.info(f"Generated metrics for {self.model_name}: {metrics}")
            
            return {
                'backtest': historical_forecasts,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Error in {self.model_name} backtesting: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def evaluate(self, actual: TimeSeries, predicted: TimeSeries) -> Dict[str, float]:
        actual_trimmed, predicted_trimmed = actual.slice_intersect(predicted), predicted.slice_intersect(actual)
        return {
            "MAPE": mape(actual_trimmed, predicted_trimmed),
            "MSE": mse(actual_trimmed, predicted_trimmed),
            "RMSE": rmse(actual_trimmed, predicted_trimmed),
        }
