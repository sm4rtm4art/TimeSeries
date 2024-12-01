"""
N-BEATS Model Implementation for Time Series Forecasting

This module implements the N-BEATS (Neural Basis Expansion Analysis for Interpretable Time Series Forecasting) model
using the Darts library. N-BEATS is a deep neural architecture based on backward and forward residual links and a very
deep stack of fully-connected layers.

Reference:
Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2020). 
N-BEATS: Neural basis expansion analysis for interpretable time series forecasting. 
International Conference on Learning Representations (ICLR).
"""

from darts.models import NBEATSModel
from backend.core.interfaces.base_model import DartsModelPredictor
from darts import TimeSeries
from typing import Dict, Union, Any
import pandas as pd
import streamlit as st
import pytorch_lightning as pl
import torch
import logging
from backend.utils.model_utils import get_training_config
from backend.domain.services.training import ModelTrainingService
import traceback
from darts.metrics import mape, rmse, mse

logger = logging.getLogger(__name__)

class NBEATSPredictor(DartsModelPredictor):
    def __init__(self):
        self.input_chunk_length = 24
        self.output_chunk_length = 12
        self.n_epochs = 100
        self.batch_size = 32
        self.optimizer_kwargs = {"lr": 1e-3}
        self.model_name = "N-BEATS"
        super().__init__()

    def _get_trainer_kwargs(self):
        training_config = get_training_config()
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        return {
            'accelerator': ModelTrainingService.determine_accelerator(),
            'precision': training_config['precision'],
            'callbacks': [
                pl.callbacks.EarlyStopping(
                    monitor="train_loss",
                    patience=10,
                    min_delta=0.000001,
                    mode="min"
                ),
                ModelTrainingService.PrintEpochResults(
                    progress_bar,
                    status_text,
                    self.n_epochs,
                    self.model_name
                )
            ],
            'enable_progress_bar': True,
            'enable_model_summary': True,
            'log_every_n_steps': 1,
            'devices': 1
        }

    def _create_model(self) -> NBEATSModel:
        return NBEATSModel(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            generic_architecture=True,
            num_stacks=10,
            num_blocks=1,
            num_layers=4,
            layer_widths=512,
            n_epochs=self.n_epochs,
            pl_trainer_kwargs=self._get_trainer_kwargs(),
            batch_size=self.batch_size,
            optimizer_kwargs=self.optimizer_kwargs
        )

    def train(self, data: TimeSeries) -> None:
        try:
            logger.info(f"Starting {self.model_name} model training...")
            self.model = self._create_model()
            self.model.fit(data, verbose=True)
            self.is_trained = True
            logger.info(f"{self.model_name} model training completed successfully")
        except Exception as e:
            logger.error(f"Error during {self.model_name} model training: {str(e)}")
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
                verbose=True
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
