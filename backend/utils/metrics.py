import traceback
import logging

import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.metrics import mae, mape, mse, rmse, smape
from typing import Dict, Union  # Make sure Union is imported

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def calculate_metrics(test_data: TimeSeries, forecast: TimeSeries) -> dict:
    try:
        # Align the forecast and test data
        test_data_trimmed, forecast_trimmed = test_data.slice_intersect(forecast), forecast.slice_intersect(test_data)

        # Check if the lengths are not zero
        if len(test_data_trimmed) == 0 or len(forecast_trimmed) == 0:
            raise ValueError("No overlapping data between test data and forecast.")

        # Calculate metrics
        metrics = {
            "MAE": mae(test_data_trimmed, forecast_trimmed),
            "MSE": mse(test_data_trimmed, forecast_trimmed),
            "RMSE": rmse(test_data_trimmed, forecast_trimmed),
            "MAPE": mape(test_data_trimmed, forecast_trimmed),
            "sMAPE": smape(test_data_trimmed, forecast_trimmed)
        }

        # Handle potential NaN or inf values
        for key, value in metrics.items():
            if np.isnan(value) or np.isinf(value):
                metrics[key] = None

        return metrics
    except Exception as e:
        print(f"Error calculating metrics: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        return {metric: None for metric in ["MAE", "MSE", "RMSE", "MAPE", "sMAPE"]}

def safe_metric(metric_func, actual: TimeSeries, forecast: TimeSeries) -> float:
    try:
        result = metric_func(actual, forecast)
        return float(result) if not np.isnan(result) else None
    except Exception as e:
        print(f"Error calculating metric {metric_func.__name__}: {str(e)}")
        return None

def calculate_metrics_for_all_models(actual: TimeSeries, forecasts: Dict[str, Dict[str, TimeSeries]]) -> Dict[str, Dict[str, float]]:
    metrics = {}
    for model_name, forecast_dict in forecasts.items():
        if 'future' in forecast_dict and forecast_dict['future'] is not None:
            forecast = forecast_dict['future']
            # Find common date range
            common_dates = set(actual.time_index) & set(forecast.time_index)
            if common_dates:
                common_dates = sorted(list(common_dates))
                actual_trimmed = actual.slice(common_dates[0], common_dates[-1])
                forecast_trimmed = forecast.slice(common_dates[0], common_dates[-1])
                
                metrics[model_name] = calculate_metrics(actual_trimmed, forecast_trimmed)
            else:
                print(f"No common dates found for {model_name}")
                metrics[model_name] = {metric: None for metric in ["MAE", "MSE", "RMSE", "MAPE", "sMAPE"]}
        else:
            print(f"No future forecast found for {model_name}")
            metrics[model_name] = {metric: None for metric in ["MAE", "MSE", "RMSE", "MAPE", "sMAPE"]}
    return metrics
