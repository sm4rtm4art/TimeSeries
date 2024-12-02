"""
Mamba Model Implementation for Time Series Forecasting

This module implements the Mamba state-space model for time series forecasting.
Mamba is a new architecture that replaces attention with selective state spaces,
offering better performance and efficiency compared to traditional transformers.

Key Features:
- Selective State Space Sequence modeling
- Linear time and memory complexity
- Efficient hardware utilization
- Support for both univariate and multivariate time series
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Union, Tuple
from darts import TimeSeries
import numpy as np
import logging
from mamba_ssm import Mamba, MambaConfig
import pytorch_lightning as pl
import pandas as pd

logger = logging.getLogger(__name__)

class MambaTimeSeriesModel(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        input_length: int = 100,
        output_length: int = 24,
        dropout: float = 0.1
    ):
        """
        Initialize Mamba model for time series forecasting.
        
        Args:
            d_model: Model dimension
            n_layers: Number of Mamba layers
            input_length: Length of input sequence
            output_length: Length of output sequence (forecast horizon)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_length = input_length
        self.output_length = output_length
        
        # Input projection
        self.input_proj = nn.Linear(1, d_model)
        
        # Mamba configuration
        self.config = MambaConfig(
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout
        )
        
        # Mamba backbone
        self.mamba = Mamba(self.config)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, 1)
        
    def forward(
        self, 
        x: torch.Tensor,
        covariates: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, input_length, features)
            covariates: Optional covariate tensor
            
        Returns:
            Predictions of shape (batch_size, output_length, features)
        """
        # Project input to model dimension
        x = self.input_proj(x)
        
        # Add covariates if provided
        if covariates is not None:
            x = torch.cat([x, covariates], dim=-1)
        
        # Pass through Mamba layers
        hidden_states = self.mamba(x)
        
        # Generate autoregressive predictions
        predictions = []
        current_input = hidden_states[:, -1:]
        
        for _ in range(self.output_length):
            output = self.mamba(current_input)
            pred = self.output_proj(output[:, -1:])
            predictions.append(pred)
            current_input = self.input_proj(pred)
            
        return torch.cat(predictions, dim=1)

class MambaPredictor:
    def __init__(
        self,
        input_length: int = 100,
        forecast_horizon: int = 24,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        n_epochs: int = 100
    ):
        """
        Mamba model predictor for time series forecasting.
        """
        self.input_length = input_length
        self.forecast_horizon = forecast_horizon
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        
        self.model = MambaTimeSeriesModel(
            input_length=input_length,
            output_length=forecast_horizon
        )
        
        self.is_trained = False
        self.model_name = "Mamba"
        
    def _prepare_data(
        self, 
        data: TimeSeries,
        past_covariates: Optional[TimeSeries] = None
    ) -> torch.Tensor:
        """Prepare data for Mamba model."""
        values = data.values()
        if past_covariates is not None:
            covariate_values = past_covariates.values()
            return torch.cat([
                torch.FloatTensor(values),
                torch.FloatTensor(covariate_values)
            ], dim=-1)
        return torch.FloatTensor(values)
        
    def train(
        self,
        data: TimeSeries,
        past_covariates: Optional[TimeSeries] = None,
        val_data: Optional[TimeSeries] = None
    ) -> None:
        """Train the Mamba model."""
        try:
            logger.info("Starting Mamba model training...")
            
            # Prepare data
            train_data = self._prepare_data(data, past_covariates)
            if val_data is not None:
                val_data = self._prepare_data(val_data, past_covariates)
            
            # Training setup
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate
            )
            criterion = nn.MSELoss()
            
            # Training loop
            for epoch in range(self.n_epochs):
                self.model.train()
                total_loss = 0
                
                for i in range(0, len(train_data) - self.input_length, self.batch_size):
                    batch_x = train_data[i:i + self.input_length]
                    batch_y = train_data[i + 1:i + self.input_length + 1]
                    
                    optimizer.zero_grad()
                    output = self.model(batch_x)
                    loss = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}/{self.n_epochs}, Loss: {total_loss:.4f}")
            
            self.is_trained = True
            logger.info("Mamba model training completed")
            
        except Exception as e:
            logger.error(f"Error in Mamba training: {str(e)}")
            raise
            
    def predict(
        self,
        horizon: int,
        data: Optional[TimeSeries] = None,
        past_covariates: Optional[TimeSeries] = None
    ) -> TimeSeries:
        """Generate predictions using the Mamba model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        try:
            # Prepare input data
            input_data = self._prepare_data(data, past_covariates)
            input_sequence = input_data[-self.input_length:]
            
            # Generate predictions
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(
                    input_sequence.unsqueeze(0),
                    horizon
                ).squeeze(0)
            
            # Convert predictions to TimeSeries
            forecast_dates = pd.date_range(
                start=data.time_index[-1] + data.freq,
                periods=horizon,
                freq=data.freq
            )
            
            return TimeSeries.from_times_and_values(
                forecast_dates,
                predictions.numpy()
            )
            
        except Exception as e:
            logger.error(f"Error in Mamba prediction: {str(e)}")
            raise
            
    def backtest(
        self,
        data: TimeSeries,
        start: Union[pd.Timestamp, float, int],
        forecast_horizon: int,
        stride: int = 1,
        retrain: bool = False
    ) -> Tuple[TimeSeries, Dict[str, float]]:
        """Perform backtesting of the model."""
        try:
            if isinstance(start, float):
                start = int(len(data) * start)
                
            historical_forecasts = []
            actual_values = []
            
            for i in range(start, len(data) - forecast_horizon, stride):
                train_data = data[:i]
                if retrain:
                    self.train(train_data)
                    
                forecast = self.predict(forecast_horizon, train_data)
                historical_forecasts.append(forecast)
                actual_values.append(data[i:i + forecast_horizon])
            
            # Combine forecasts
            combined_forecast = TimeSeries.from_series(
                pd.concat([f.pd_series() for f in historical_forecasts])
            )
            
            # Calculate metrics
            metrics = {
                'MAPE': mape(actual_values, historical_forecasts),
                'RMSE': rmse(actual_values, historical_forecasts),
                'MAE': mae(actual_values, historical_forecasts)
            }
            
            return combined_forecast, metrics
            
        except Exception as e:
            logger.error(f"Error in Mamba backtesting: {str(e)}")
            raise
