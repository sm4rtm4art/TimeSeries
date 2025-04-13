"""AutoARIMA Model Implementation

This module provides a wrapper for the Darts AutoARIMA model, which automatically
selects the best ARIMA model parameters based on statistical criteria.
"""

import logging
import warnings

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.metrics import mae, mape, rmse
from darts.models import AutoARIMA
from threadpoolctl import threadpool_limits  # Import for parallelism control

# Suppress sklearn and statsmodels warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning,
                        module="statsmodels")

logger = logging.getLogger(__name__)


class ARIMAPredictor:
    def __init__(
        self,
        start_p: int = 0,
        start_q: int = 0,
        max_p: int = 2,
        max_q: int = 2,
        d: int = 1,
        seasonal: bool = False,  # Turn off seasonal by default (much faster)
        m: int = 0,  # Set m=0 when seasonal=False to avoid warnings
        suppress_warnings: bool = True,  # Suppress statsmodels warnings
        max_iter: int = 50,  # Limit iterations
        information_criterion: str = "aic",  # Use AIC (faster than BIC)
        stepwise: bool = True,  # Use stepwise algorithm (faster)
        approximation: bool = True,  # Use approximation (faster)
        timeout: int = 30,  # Timeout in seconds
        limit_parallelism: bool = True,  # NEW: Control parallelism
    ):
        """Initialize AutoARIMA model with conservative defaults to prevent infinite loops.

        Args:
            start_p: Starting value of p for the ARIMA model.
            start_q: Starting value of q for the ARIMA model.
            max_p: Maximum value of p for the ARIMA model.
            max_q: Maximum value of q for the ARIMA model.
            d: Differencing order (default=1).
            seasonal: Whether to fit a seasonal model (default=False for speed).
            m: Seasonal period. Set to 0 when seasonal=False.
            suppress_warnings: Whether to suppress statsmodels warnings.
            max_iter: Maximum number of iterations for parameter fitting.
            information_criterion: Criterion for model selection ('aic', 'bic', etc.).
            stepwise: Whether to use the stepwise algorithm for faster fitting.
            approximation: Whether to use approximation for faster fitting.
            timeout: Timeout in seconds for model fitting (prevents infinite loops).
            limit_parallelism: Whether to limit BLAS parallelism for faster processing.
        """
        self.model_name = "ARIMA"
        self.limit_parallelism = limit_parallelism
        self.model = AutoARIMA(
            start_p=start_p,
            start_q=start_q,
            max_p=max_p,
            max_q=max_q,
            d=d,
            seasonal=seasonal,
            m=m,
            suppress_warnings=suppress_warnings,
            max_iter=max_iter,
            information_criterion=information_criterion,
            stepwise=stepwise,
            approximation=approximation,
            timeout=timeout,
        )
        self.is_trained = False

    def train(
        self,
        data: TimeSeries,
        past_covariates: TimeSeries = None,
        future_covariates: TimeSeries = None,
        verbose: bool = True,
    ) -> None:
        """Train the ARIMA model.

        Args:
            data: Training time series data.
            past_covariates: Optional past covariates (not used for ARIMA).
            future_covariates: Optional future covariates (not used for ARIMA).
            verbose: Whether to print training progress.
        """
        try:
            logger.info("Training ARIMA model...")

            # AutoARIMA can be sensitive to data types
            # Use numpy's dtype to avoid type errors
            data = data.astype(np.dtype("float32"))

            # Apply parallelism limits if requested
            # Based on https://njodell.com/?p=74 - can speed up ARIMA by 3.7x
            if self.limit_parallelism:
                logger.info("Limiting BLAS parallelism to speed up ARIMA fitting")
                with threadpool_limits(limits=1, user_api="blas"):
                    # Fit the model with limited parallelism
                    self.model.fit(series=data)
            else:
                # Fit the model normally
                self.model.fit(series=data)

            self.is_trained = True
            logger.info("ARIMA model training completed successfully")
        except Exception as e:
            logger.error(f"Error during ARIMA model training: {str(e)}")
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
            series: Optional series to use (instead of training data).
            past_covariates: Optional past covariates (not used for ARIMA).
            future_covariates: Optional future covariates (not used for ARIMA).

        Returns:
            TimeSeries: Forecasted values.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        try:
            # Apply parallelism limits if requested
            if self.limit_parallelism:
                with threadpool_limits(limits=1, user_api="blas"):
                    forecast = self.model.predict(n=n, series=series)
            else:
                forecast = self.model.predict(n=n, series=series)
            return forecast
        except Exception as e:
            logger.error(f"Error during ARIMA prediction: {str(e)}")
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
                    Note: ARIMA models in Darts require retrain=True.
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
            logger.error(f"Error during ARIMA backtesting: {str(e)}")
            raise
