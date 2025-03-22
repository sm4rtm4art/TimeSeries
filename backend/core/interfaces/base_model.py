import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, mse, rmse

logger = logging.getLogger(__name__)


class TimeSeriesPredictor(ABC):
    """Base class for all time series prediction models."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.trainer_params = self._get_hardware_config()

    def _get_hardware_config(self) -> dict[str, Any]:
        """Determine the best hardware configuration."""
        if torch.cuda.is_available():
            return {"accelerator": "gpu", "precision": "32-true"}
        elif torch.backends.mps.is_available():
            # Some models don't support MPS yet, they should override this
            return {"accelerator": "mps", "precision": "32-true"}
        else:
            return {"accelerator": "cpu", "precision": "32-true"}

    def train(self, data: TimeSeries, **kwargs) -> None:
        """Train the model on the provided data."""
        try:
            # 1. Initialize components if first time
            if self.scaler is None:
                self.scaler = Scaler()

            # 2. Prepare data (convert to float32 and scale)
            data_float32 = data.astype(np.float32)
            scaled_data = self.scaler.fit_transform(data_float32)

            # 3. Initialize and train the model
            logger.info(f"Training {self.model_name} on {self.trainer_params['accelerator']}")
            self._train_model(scaled_data, **kwargs)
            self.is_trained = True

            logger.info(f"{self.model_name} trained successfully")

        except Exception as e:
            logger.error(f"Error training {self.model_name}: {str(e)}")
            raise

    @abstractmethod
    def _train_model(self, scaled_data: TimeSeries, **kwargs) -> None:
        """Model-specific training implementation."""
        pass

    def predict(self, horizon: int) -> TimeSeries:
        """Generate predictions for the given horizon."""
        if not self.is_trained:
            raise ValueError(f"{self.model_name} must be trained before prediction")

        try:
            forecast = self._generate_forecast(horizon)
            return self.scaler.inverse_transform(forecast)
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise

    @abstractmethod
    def _generate_forecast(self, horizon: int) -> TimeSeries:
        """Model-specific forecast implementation."""
        pass

    def backtest(
        self,
        data: TimeSeries,
        start: float,
        forecast_horizon: int,
        stride: int = 1,
        retrain: bool = False,
    ) -> dict[str, Any]:
        """Perform backtesting on historical data."""
        try:
            scaled_data = self.scaler.transform(data.astype(np.float32))
            historical_forecasts = self._generate_historical_forecasts(
                scaled_data,
                start,
                forecast_horizon,
                stride,
                retrain,
            )

            return {
                "backtest": self.scaler.inverse_transform(historical_forecasts),
                "metrics": self._calculate_metrics(data, historical_forecasts),
            }
        except Exception as e:
            logger.error(f"Error in backtesting: {str(e)}")
            raise

    @abstractmethod
    def _generate_historical_forecasts(
        self,
        series: TimeSeries,
        start: float,
        forecast_horizon: int,
        stride: int,
        retrain: bool,
    ) -> TimeSeries:
        """Model-specific historical forecasts implementation."""
        pass

    def _calculate_metrics(self, actual: TimeSeries, predicted: TimeSeries) -> dict[str, float]:
        """Calculate forecast accuracy metrics."""
        try:
            return {
                "MAPE": float(mape(actual, predicted)),
                "RMSE": float(rmse(actual, predicted)),
                "MSE": float(mse(actual, predicted)),
            }
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise
