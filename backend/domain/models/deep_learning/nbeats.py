"""N-BEATS Model Implementation for Time Series Forecasting

This module implements the N-BEATS (Neural Basis Expansion Analysis for Interpretable Time Series Forecasting) model
using the Darts library. N-BEATS is a deep neural architecture based on backward and forward residual links and a very
deep stack of fully-connected layers.

Reference:
Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2020).
N-BEATS: Neural basis expansion analysis for interpretable time series forecasting.
International Conference on Learning Representations (ICLR).
"""

import logging
import time
from typing import Any

import pytorch_lightning as pl
from darts import TimeSeries
from darts.models import NBEATSModel

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
            f"Training N-BEATS: Epoch {current_epoch + 1}/{self.total_epochs}, Loss: {loss:.4f}",
        )


class NBEATSPredictor(TimeSeriesPredictor):
    def __init__(self, model_name: str = "N-BEATS"):
        super().__init__(model_name)
        self._initialize_model()

    def _get_hardware_config(self) -> dict[str, Any]:
        """Override hardware config for N-BEATS."""
        config = super()._get_hardware_config()

        # For performance comparison, log both CPU and MPS options
        if config["accelerator"] == "mps":
            logger.warning("MPS detected but not supported well by N-BEATS. Falling back to CPU.")

            # Option for users to force MPS if they want to try it
            import os
            if os.environ.get("FORCE_MPS_FOR_NBEATS", "0") == "1":
                logger.warning("FORCE_MPS_FOR_NBEATS=1: Using MPS despite performance concerns")
                return {"accelerator": "mps", "precision": "32-true"}
            return {"accelerator": "cpu", "precision": "32-true"}
        return config

    def _initialize_model(self):
        try:
            # Get hardware config
            hw_config = self._get_hardware_config()
            is_mps = hw_config.get("accelerator") == "mps"

            # Adjust model parameters to be more efficient on MPS if using MPS
            model_params = {
                "input_chunk_length": 24,
                "output_chunk_length": 12,
                "generic_architecture": True,
                # Reduce complexity if using MPS for better performance
                "num_stacks": 10 if is_mps else 30,
                "num_blocks": 1,
                "num_layers": 4,
                "layer_widths": 128 if is_mps else 256,
                "batch_size": 64 if is_mps else 32,  # Larger batch size can help GPU utilization
                "n_epochs": 100,
                "pl_trainer_kwargs": {
                    **self.trainer_params,
                    # Add MPS-specific optimizations if needed
                    **({"gradient_clip_val": 1.0} if is_mps else {})
                },
            }

            self.model = NBEATSModel(**model_params)
            logger.info(f"N-BEATS model initialized with config: {model_params}")
        except Exception as e:
            logger.error(f"Error initializing N-BEATS model: {str(e)}")
            raise

    def _train_model(self, scaled_data: TimeSeries, **kwargs) -> None:
        """Train the N-BEATS model."""
        # Add performance profiling
        start_time = time.time()
        accelerator = self.trainer_params.get("accelerator", "cpu")

        self.model.fit(scaled_data, verbose=True)

        elapsed = time.time() - start_time
        logger.info(f"N-BEATS training completed in {elapsed:.2f} seconds using {accelerator}")

    def _generate_forecast(self, horizon: int) -> TimeSeries:
        """Generate forecast using the trained model."""
        return self.model.predict(n=horizon)

    def _generate_historical_forecasts(
        self,
        series: TimeSeries,
        start: float,
        forecast_horizon: int,
        stride: int,
        retrain: bool,
    ) -> TimeSeries:
        """Generate historical forecasts for backtesting."""
        return self.model.historical_forecasts(
            series=series,
            start=start,
            forecast_horizon=forecast_horizon,
            stride=stride,
            retrain=retrain,
            verbose=True,
        )
