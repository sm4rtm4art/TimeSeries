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
from typing import Optional, Dict, Union, Tuple, Any
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
from darts.metrics import mape, rmse, mse
from backend.utils.scaling import scale_data, inverse_scale
from ....models.base_model import BasePredictor

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
        progress = (current_epoch + 1) / self.total_epochs
        self.progress_bar.progress(progress)
        self.status_text.text(f"Training Progress: {int(progress * 100)}%")


class TSMixerPredictor(BasePredictor):
    def __init__(self):
        super().__init__(model_name="TSMixer")
        self.input_chunk_length = 24
        self.output_chunk_length = 12
        self.hidden_size = 64
        self.n_epochs = 100
        self.dropout = 0.1
        self.random_state = 42

    def _create_model(self) -> Any:
        """Create and return a TSMixer model instance."""
        return TSMixerModel(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            random_state=self.random_state,
            pl_trainer_kwargs={
                'accelerator': determine_accelerator(),
                'enable_progress_bar': False,
                'precision': '32-true',
                'max_epochs': self.n_epochs
            }
        )

    def _train_model(self, data: TimeSeries) -> None:
        """Train the TSMixer model."""
        try:
            # Scale the data using the full dataset
            scaled_data = self.scaler.fit_transform(data)
            
            # Train the model
            self.model.fit(scaled_data)
            
        except Exception as e:
            logger.error(f"Error training TSMixer: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _generate_forecast(self, n: int, series: TimeSeries) -> TimeSeries:
        """Generate forecast using the trained model."""
        try:
            # Scale using the fitted scaler
            scaled_data = self.scaler.transform(series)
            
            # Generate prediction
            scaled_forecast = self.model.predict(n=n, series=scaled_data)
            
            # Inverse transform the prediction
            return self.scaler.inverse_transform(scaled_forecast)
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise

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
        stride: int = 1,
        retrain: bool = False
    ) -> Dict[str, Union[TimeSeries, Dict[str, float]]]:
        """Perform backtesting."""
        try:
            # Scale the data
            scaled_data = self.scaler.fit_transform(data.astype(np.float32))
            
            # Perform historical forecasts
            scaled_forecasts = self.model.historical_forecasts(
                series=scaled_data,
                start=start,
                forecast_horizon=forecast_horizon,
                stride=stride,
                retrain=False,  # Force no retraining
                verbose=False,
                show_warnings=False
            )
            
            # Inverse transform the results
            forecasts = self.scaler.inverse_transform(scaled_forecasts)
            
            # Calculate metrics
            actual = data[forecasts.start_time():forecasts.end_time()]
            metrics = {
                'MAPE': float(mape(actual, forecasts)),
                'RMSE': float(rmse(actual, forecasts)),
                'MSE': float(mse(actual, forecasts))
            }
            
            return {
                'backtest': forecasts,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def evaluate(self, actual: TimeSeries, predicted: TimeSeries) -> Dict[str, float]:
        # Ensure the time ranges match
        actual_trimmed, predicted_trimmed = actual.slice_intersect(predicted), predicted.slice_intersect(actual)

        return {
            "MAPE": mape(actual_trimmed, predicted_trimmed),
            "MSE": mse(actual_trimmed, predicted_trimmed),
            "RMSE": rmse(actual_trimmed, predicted_trimmed),
        }

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

