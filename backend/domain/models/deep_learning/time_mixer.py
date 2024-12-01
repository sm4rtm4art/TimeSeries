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
from backend.core.interfaces.base_model import DartsModelPredictor
from backend.utils.model_utils import get_training_config

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


class TSMixerPredictor(DartsModelPredictor):
    def __init__(self):
        self.model_name = "TSMixer"
        self.hidden_size = 64
        self.dropout = 0.1
        self.n_epochs = 100
        self.batch_size = 32
        self.optimizer_kwargs = {"lr": 1e-3}
        super().__init__()  # This will set input_chunk_length and output_chunk_length

    def _create_model(self) -> Any:
        training_config = get_training_config()
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        trainer_kwargs = {
            'accelerator': training_config['accelerator'],
            'precision': training_config['precision'],
            'callbacks': [
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
            'enable_progress_bar': True,
            'enable_model_summary': True,
            'log_every_n_steps': 1
        }
        
        return TSMixerModel(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            optimizer_kwargs=self.optimizer_kwargs,
            pl_trainer_kwargs=trainer_kwargs
        )

    def train(self, data: TimeSeries) -> None:
        try:
            logger.info(f"Training {self.model_name} model")
            training_config = get_training_config()
            # Convert data to appropriate dtype before scaling
            data = data.astype(training_config['force_dtype'])
            scaled_data = self.scaler.fit_transform(data)
            self.model.fit(scaled_data)
            self.is_trained = True
            logger.info(f"{self.model_name} model trained successfully")
        except Exception as e:
            logger.error(f"Error training {self.model_name} model: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def predict(self, n: int) -> TimeSeries:
        """
        Generate predictions for n steps ahead.
        
        Args:
            n (int): Number of steps to forecast
            
        Returns:
            TimeSeries: Forecasted values
        """
        try:
            if not self.is_trained:
                raise ValueError(f"{self.model_name} must be trained before prediction")
            forecast = self.model.predict(n=n)
            return self.scaler.inverse_transform(forecast)
        except Exception as e:
            logger.error(f"Error in {self.model_name} prediction: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def backtest(
        self,
        data: TimeSeries,
        start: Union[pd.Timestamp, float],
        forecast_horizon: int,
        stride: int = 1
    ) -> Dict[str, Union[TimeSeries, Dict[str, float]]]:
        try:
            if not self.is_trained:
                raise ValueError(f"{self.model_name} must be trained before backtesting")
                
            logger.info(f"Starting {self.model_name} backtesting")
            
            scaled_data = self.scaler.transform(data)
            historical_forecasts = self.model.historical_forecasts(
                series=scaled_data,
                start=start,
                forecast_horizon=forecast_horizon,
                stride=stride,
                retrain=False,
                verbose=False
            )
            
            forecasts = self.scaler.inverse_transform(historical_forecasts)
            actual_data = data[forecasts.start_time():forecasts.end_time()]
            
            metrics = {
                'MAPE': float(mape(actual_data, forecasts)),
                'RMSE': float(rmse(actual_data, forecasts)),
                'MSE': float(mse(actual_data, forecasts))
            }
            
            return {
                'backtest': forecasts,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Error in {self.model_name} backtesting: {str(e)}")
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

