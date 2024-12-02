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
from typing import Dict, Union, Any
import logging
import torch
import pytorch_lightning as pl
from darts import TimeSeries
from darts.models import TiDEModel
from pytorch_lightning.callbacks import EarlyStopping
from backend.core.interfaces.base_model import TimeSeriesPredictor

logger = logging.getLogger(__name__)

class PrintCallback(pl.Callback):
    def __init__(self, progress_bar, status_text, total_epochs: int):
        super().__init__()
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.total_epochs = total_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        loss = trainer.callback_metrics.get("train_loss", 0).item()
        progress = (current_epoch + 1) / self.total_epochs
        self.progress_bar.progress(progress)
        self.status_text.text(
            f"Training TiDE model: Epoch {current_epoch + 1}/{self.total_epochs}, Loss: {loss:.4f}"
        )

class TiDEPredictor(TimeSeriesPredictor):
    def __init__(self, model_name: str = "TiDE"):
        super().__init__(model_name)
        self._initialize_model()

    def _get_hardware_config(self) -> Dict[str, Any]:
        """Override hardware config for TiDE."""
        config = super()._get_hardware_config()
        if config['accelerator'] == 'mps':
            logger.warning("MPS detected but not supported by TiDE. Falling back to CPU.")
            return {'accelerator': 'cpu', 'precision': '32-true'}
        return config

    def _initialize_model(self):
        try:
            # Default configuration
            model_params = {
                'input_chunk_length': 24,
                'output_chunk_length': 12,
                'hidden_size': 64,
                'dropout': 0.1,
                'batch_size': 32,
                'n_epochs': 100,
                'model_name': self.model_name,
                'force_reset': True,
                'pl_trainer_kwargs': self.trainer_params
            }
            
            self.model = TiDEModel(**model_params)
            logger.info(f"TiDE model initialized with config: {model_params}")
        except Exception as e:
            logger.error(f"Error initializing TiDE model: {str(e)}")
            raise

    def _train_model(self, scaled_data: TimeSeries, **kwargs) -> None:
        """Train the TiDE model."""
        self.model.fit(scaled_data, verbose=True)

    def _generate_forecast(self, horizon: int) -> TimeSeries:
        """Generate forecast using the trained model."""
        return self.model.predict(n=horizon)

    def _generate_historical_forecasts(
        self, series: TimeSeries, start: float, 
        forecast_horizon: int, stride: int, retrain: bool
    ) -> TimeSeries:
        """Generate historical forecasts for backtesting."""
        return self.model.historical_forecasts(
            series=series,
            start=start,
            forecast_horizon=forecast_horizon,
            stride=stride,
            retrain=retrain,
            verbose=True
        )
