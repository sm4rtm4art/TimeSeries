"""Prophet Model Implementation for Time Series Forecasting

This module implements the Prophet model for time series forecasting using the Darts library.
Prophet is a procedure for forecasting time series data based on an additive model where non-linear
trends are fit with yearly, weekly, and daily seasonality, plus holiday effects.

Key Features:
- Implements the Prophet model from Darts
- Provides methods for training, prediction, historical forecasts, and backtesting
- Includes evaluation metrics for model performance assessment
"""

import logging

from darts import TimeSeries
from darts.models import Prophet

from backend.core.interfaces.base_model import TimeSeriesPredictor

logger = logging.getLogger(__name__)


class ProphetModel(TimeSeriesPredictor):
    def __init__(self, model_name: str = "Prophet"):
        super().__init__(model_name)
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the Prophet model with configuration."""
        try:
            model_params = {
                "seasonality_mode": "multiplicative",
                "yearly_seasonality": True,
                "weekly_seasonality": False,
                "daily_seasonality": False,
                "growth": "linear",
                "changepoint_prior_scale": 0.05,
                "seasonality_prior_scale": 10.0,
            }

            self.model = Prophet(**model_params)
            logger.info("Prophet model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Prophet model: {str(e)}")
            raise

    def _train_model(self, scaled_data: TimeSeries, **kwargs) -> None:
        """Train the Prophet model."""
        self.model.fit(scaled_data)

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
            retrain=True,
            verbose=True,
        )
