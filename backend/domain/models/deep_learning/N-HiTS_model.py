"""N-HiTS (Neural Hierarchical Interpolation for Time Series) Model Implementation

Key features:
- Hierarchical interpolation for multi-scale patterns
- Efficient long-horizon forecasting
- Support for both univariate and multivariate data
"""

import logging

import numpy as np
import pytorch_lightning as pl
import streamlit as st
import torch
from darts import TimeSeries
from darts.metrics import mae, mape, rmse
from darts.models import NHiTSModel

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
            f"Training N-HiTS model: Epoch {current_epoch + 1}/{self.total_epochs}, Loss: {loss:.4f}",
        )


class NHiTSPredictor:
    def __init__(
        self,
        input_chunk_length: int = 24,
        output_chunk_length: int = 12,
        num_stacks: int = 3,
        num_blocks: int = 1,
        num_layers: int = 2,
        layer_widths: int = 512,
        n_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
    ):
        """Initialize N-HiTS model.

        Args:
            input_chunk_length: Length of input sequences
            output_chunk_length: Length of output sequences (forecast horizon)
            num_stacks: Number of stacks in the architecture
            num_blocks: Number of blocks per stack
            num_layers: Number of layers per block
            layer_widths: Width of the layers
            n_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimization

        """
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.n_epochs = n_epochs
        self.model = None
        self.scaler = None
        self.is_trained = False

        # Determine device (GPU/MPS/CPU)
        self.device = self._determine_device()

        # Create progress tracking components
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()

        # Initialize model
        self.model = NHiTSModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            num_stacks=num_stacks,
            num_blocks=num_blocks,
            num_layers=num_layers,
            layer_widths=layer_widths,
            activation="ReLU",
            batch_size=batch_size,
            n_epochs=n_epochs,
            optimizer_kwargs={"lr": learning_rate},
            pl_trainer_kwargs={
                "accelerator": self.device,
                "callbacks": [PrintCallback(self.progress_bar, self.status_text, n_epochs)],
                "enable_progress_bar": False,
            },
        )

    def _determine_device(self) -> str:
        """Determine the available hardware accelerator."""
        if torch.cuda.is_available():
            return "gpu"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def train(
        self,
        data: TimeSeries,
        past_covariates: TimeSeries | None = None,
        future_covariates: TimeSeries | None = None,
        verbose: bool = True,
    ) -> None:
        """Train the N-HiTS model.

        Args:
            data: Training time series data
            past_covariates: Optional past covariates
            future_covariates: Optional future covariates
            verbose: Whether to print training progress

        """
        try:
            logger.info("Starting N-HiTS model training...")

            # Convert data to float32
            data = self._prepare_data(data)

            # Train the model
            self.model.fit(
                series=data,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
                verbose=verbose,
            )

            self.is_trained = True
            logger.info("N-HiTS model training completed successfully")

        except Exception as e:
            logger.error(f"Error during N-HiTS model training: {str(e)}")
            raise

    def _prepare_data(self, data: TimeSeries) -> TimeSeries:
        """Prepare data for training/prediction."""
        return data.astype(np.float32)

    def predict(
        self,
        n: int,
        series: TimeSeries | None = None,
        past_covariates: TimeSeries | None = None,
        future_covariates: TimeSeries | None = None,
    ) -> TimeSeries:
        """Generate predictions using the trained model.

        Args:
            n: Forecast horizon
            series: Optional series to use (instead of training data)
            past_covariates: Optional past covariates
            future_covariates: Optional future covariates

        Returns:
            TimeSeries: Forecasted values

        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        try:
            forecast = self.model.predict(
                n=n,
                series=series,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
            )
            return forecast
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    def backtest(
        self,
        data: TimeSeries,
        start: pd.Timestamp | float | int,
        forecast_horizon: int,
        stride: int = 1,
        retrain: bool = False,
        verbose: bool = False,
    ) -> tuple[TimeSeries, dict[str, float]]:
        """Perform backtesting of the model.

        Args:
            data: Complete time series
            start: Start point for backtesting
            forecast_horizon: Number of steps to forecast in each iteration
            stride: Number of steps between forecast points
            retrain: Whether to retrain the model at each iteration
            verbose: Whether to print progress

        Returns:
            Tuple of (historical forecasts, metrics dictionary)

        """
        try:
            historical_forecasts = self.model.historical_forecasts(
                series=data,
                start=start,
                forecast_horizon=forecast_horizon,
                stride=stride,
                retrain=retrain,
                verbose=verbose,
            )

            # Calculate metrics
            actual_data = data.slice(start, data.end_time())
            metrics = {
                "MAPE": mape(actual_data, historical_forecasts),
                "RMSE": rmse(actual_data, historical_forecasts),
                "MAE": mae(actual_data, historical_forecasts),
            }

            return historical_forecasts, metrics

        except Exception as e:
            logger.error(f"Error during backtesting: {str(e)}")
            raise
