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

from typing import Optional, Dict, Union, Tuple, Any
import logging
import torch
import pytorch_lightning as pl
from darts import TimeSeries
from darts.models import TSMixerModel
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
            f"Training TSMixer: Epoch {current_epoch + 1}/{self.total_epochs}, Loss: {loss:.4f}"
        )

class TSMixerPredictor(TimeSeriesPredictor):
    def __init__(self, model_name: str = "TSMixer"):
        super().__init__(model_name)
        self._initialize_model()

    def _get_hardware_config(self) -> Dict[str, Any]:
        """Override hardware config for TSMixer."""
        config = super()._get_hardware_config()
        if config['accelerator'] == 'mps':
            logger.warning("MPS detected but not supported by TSMixer. Falling back to CPU.")
            return {'accelerator': 'cpu', 'precision': '32-true'}
        return config

    def _initialize_model(self):
        try:
            model_params = {
                'input_chunk_length': 24,
                'output_chunk_length': 12,
                'hidden_size': 64,
                'dropout': 0.1,
                'batch_size': 32,
                'n_epochs': 100,
                'pl_trainer_kwargs': self.trainer_params
            }
            
            self.model = TSMixerModel(**model_params)
            logger.info(f"TSMixer model initialized with config: {model_params}")
        except Exception as e:
            logger.error(f"Error initializing TSMixer model: {str(e)}")
            raise

    def _train_model(self, scaled_data: TimeSeries, **kwargs) -> None:
        """Train the TSMixer model."""
        self.model.fit(scaled_data, verbose=True)

    def _generate_forecast(self, horizon: int) -> TimeSeries:
        """Generate forecast using the trained model."""
        return self.model.predict(n=horizon)

    def _generate_historical_forecasts(
        self, 
        series: TimeSeries,
        start: float,
        forecast_horizon: int,
        stride: int,
        retrain: bool
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

