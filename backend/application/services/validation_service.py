import logging

import numpy as np
from darts import TimeSeries

from backend.core.interfaces.base_model import TimeSeriesPredictor

logger = logging.getLogger(__name__)


class ValidationService:
    @staticmethod
    def perform_cross_validation(
        model: TimeSeriesPredictor,
        data: TimeSeries,
        num_folds: int = 5,
        horizon: int = 24,
        overlap_size: int = 0,
    ) -> dict[str, list[float] | dict[str, float]]:
        """Perform time series cross-validation with optional overlapping folds."""
        total_length = len(data)
        fold_size = (total_length - horizon) // num_folds

        if fold_size <= horizon:
            raise ValueError("Data length insufficient for specified parameters")

        metrics_per_fold = []
        forecasts = []

        for fold in range(num_folds):
            try:
                # Calculate fold boundaries
                start_idx = fold * (fold_size - overlap_size)
                end_idx = start_idx + fold_size + horizon

                if end_idx > total_length:
                    break

                # Split data for this fold
                train_data = data[: start_idx + fold_size]
                test_data = data[start_idx + fold_size : end_idx]

                # Train model on this fold
                model.train(train_data)

                # Generate forecast
                forecast = model.predict(len(test_data))
                forecasts.append(forecast)

                # Calculate metrics for this fold
                fold_metrics = {
                    "MAPE": float(model.evaluate(test_data, forecast)["MAPE"]),
                    "RMSE": float(model.evaluate(test_data, forecast)["RMSE"]),
                    "MSE": float(model.evaluate(test_data, forecast)["MSE"]),
                }

                metrics_per_fold.append(fold_metrics)
                logger.info(f"Fold {fold + 1} metrics: {fold_metrics}")

            except Exception as e:
                logger.error(f"Error in fold {fold + 1}: {str(e)}")
                continue

        # Calculate aggregate statistics
        aggregate_metrics = {}
        for metric in ["MAPE", "RMSE", "MSE"]:
            values = [m[metric] for m in metrics_per_fold]
            aggregate_metrics[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

        return {
            "fold_metrics": metrics_per_fold,
            "aggregate_metrics": aggregate_metrics,
            "forecasts": forecasts,
        }
