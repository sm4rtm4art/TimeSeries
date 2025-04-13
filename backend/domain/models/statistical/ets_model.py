"""Exponential Smoothing (ETS) Model Implementation

This module provides a wrapper for the Darts ExponentialSmoothing model,
which implements standard Holt-Winters exponential smoothing.
"""

import logging
import warnings

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.metrics import mae, mape, rmse
from darts.models import ExponentialSmoothing
from darts.models.forecasting.exponential_smoothing import (ModelMode,
                                                            SeasonalityMode)
from threadpoolctl import threadpool_limits  # Import for parallelism control

# Suppress scikit-learn FutureWarnings which come from the Darts dependency
warnings.filterwarnings("ignore", category=FutureWarning,
                        message="'force_all_finite' was renamed to 'ensure_all_finite'")

logger = logging.getLogger(__name__)


class ETSPredictor:
    def __init__(
        self,
        seasonal_periods: int = 12,
        # Simple model with no damped_trend (which causes errors)
        trend: ModelMode | None = ModelMode.ADDITIVE,
        seasonal: SeasonalityMode | None = SeasonalityMode.ADDITIVE,
        limit_parallelism: bool = True,  # NEW: Control parallelism
    ):
        """Initialize Exponential Smoothing model.

        Args:
            seasonal_periods: Number of time steps in a seasonal period.
            trend: Type of trend component (add, mul, or None).
            seasonal: Type of seasonal component (add, mul, or None).
            limit_parallelism: Whether to limit BLAS parallelism for faster processing.
        """
        self.model_name = "ETS"
        self.limit_parallelism = limit_parallelism

        # Create a simple model - avoid damped_trend parameter
        self.model = ExponentialSmoothing(
            seasonal_periods=seasonal_periods,
            trend=trend,
            seasonal=seasonal,
        )
        self.is_trained = False

    def train(
        self,
        data: TimeSeries,
        past_covariates: TimeSeries = None,
        future_covariates: TimeSeries = None,
        verbose: bool = True,  # Kept for interface compatibility
    ) -> None:
        """Train the ETS model.

        Args:
            data: Training time series data.
            past_covariates: Optional past covariates (not used for ETS).
            future_covariates: Optional future covariates (not used for ETS).
            verbose: Whether to print training progress (not used by ETS).
        """
        try:
            logger.info("Training ETS model...")

            # Convert data to float32 using proper dtype
            data = data.astype(np.dtype("float32"))

            # Apply parallelism limits if requested
            if self.limit_parallelism:
                logger.info("Limiting BLAS parallelism to speed up ETS fitting")
                with threadpool_limits(limits=1, user_api="blas"):
                    # Fit the model with limited parallelism
                    self.model.fit(series=data)
            else:
                # Fit the model normally
                self.model.fit(series=data)

            self.is_trained = True
            logger.info("ETS model training completed successfully")
        except Exception as e:
            logger.error(f"Error during ETS model training: {str(e)}")
            raise

    def predict(
        self,
        n: int,
        series: TimeSeries = None,
        past_covariates: TimeSeries = None,
        future_covariates: TimeSeries = None,
    ) -> TimeSeries:
        """Generate predictions using the trained model.

        Args:
            n: Forecast horizon.
            series: Optional series to use (ignored by ETS, uses training data).
            past_covariates: Optional past covariates (not used for ETS).
            future_covariates: Optional future covariates (not used for ETS).

        Returns:
            TimeSeries: Forecasted values.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        try:
            # ExponentialSmoothing doesn't accept a series parameter
            # Apply parallelism limits if requested
            if self.limit_parallelism:
                with threadpool_limits(limits=1, user_api="blas"):
                    forecast = self.model.predict(n=n)
            else:
                forecast = self.model.predict(n=n)
            return forecast
        except Exception as e:
            logger.error(f"Error during ETS prediction: {str(e)}")
            raise

    def backtest(
        self,
        data: TimeSeries,
        start: pd.Timestamp | int,
        forecast_horizon: int,
        stride: int = 1,
        retrain: bool = False,  # Parameter kept for interface compatibility
        verbose: bool = False,
    ) -> dict[str, TimeSeries | dict[str, float]]:
        """Perform backtesting of the model.

        Args:
            data: Complete time series.
            start: Start point for backtesting.
            forecast_horizon: Number of steps to forecast in each iteration.
            stride: Number of steps between forecast points.
            retrain: Whether to retrain the model at each iteration.
                    Note: ETS models in Darts require retrain=True.
            verbose: Whether to print progress.

        Returns:
            Dictionary with backtest results and metrics.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before backtesting")

        try:
            # Apply parallelism limits if requested
            if self.limit_parallelism:
                with threadpool_limits(limits=1, user_api="blas"):
                    # Perform backtesting with limited parallelism
                    historical_forecasts = self.model.historical_forecasts(
                        series=data,
                        start=start,
                        forecast_horizon=forecast_horizon,
                        stride=stride,
                        retrain=True,  # Force retrain=True regardless of input parameter
                    )
            else:
                # Perform backtesting normally
                historical_forecasts = self.model.historical_forecasts(
                    series=data,
                    start=start,
                    forecast_horizon=forecast_horizon,
                    stride=stride,
                    retrain=True,  # Force retrain=True regardless of input parameter
                )

            # Calculate metrics
            if isinstance(historical_forecasts, list):
                raise NotImplementedError(
                    "Metric calculation for list forecasts not implemented yet.",
                )

            try:
                # SIMPLIFY: Don't use slice() at all to avoid timestamp type issues
                # Instead, directly compare forecasts with actual values over the same period
                # This is much more reliable than slicing with timestamps

                # Get the time indices from the forecasts
                forecast_indices = historical_forecasts.time_index

                # Extract data values for these same indices (if they exist in data)
                actual_indices = [idx for idx in forecast_indices if idx in data.time_index]

                if len(actual_indices) > 0:
                    # Create actual data series using the same indices as the forecast
                    actual_values = data.loc[actual_indices]

                    # Calculate metrics only if we have matching data points
                    metrics = {
                        "MAPE": float(mape(actual_values, historical_forecasts)),
                        "RMSE": float(rmse(actual_values, historical_forecasts)),
                        "MAE": float(mae(actual_values, historical_forecasts)),
                    }
                else:
                    # Not enough overlapping points for valid metrics
                    logger.warning("No overlapping time points for metric calculation")
                    metrics = {
                        "MAPE": 999.99,
                        "RMSE": 999.99,
                        "MAE": 999.99,
                    }
            except Exception as e:
                logger.error(f"Error calculating metrics: {str(e)}")
                # Provide default metrics if calculation fails
                metrics = {
                    "MAPE": 999.99,
                    "RMSE": 999.99,
                    "MAE": 999.99,
                }

            # Return results in dictionary format
            return {
                "backtest": historical_forecasts,
                "metrics": metrics,
            }

        except Exception as e:
            logger.error(f"Error during ETS backtesting: {str(e)}")
            raise
