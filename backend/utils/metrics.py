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
    logger.info(f"Actual data length: {len(actual)}, time index: {actual.time_index}")
    
    for model_name, forecast_dict in forecasts.items():
        if 'backtest' in forecast_dict:
            backtest = forecast_dict['backtest']
            if backtest is not None and isinstance(backtest, TimeSeries):
                logger.info(f"{model_name} backtest length: {len(backtest)}, time index: {backtest.time_index}")
                
                # Ensure that actual and backtest have overlapping time periods
                common_dates = actual.time_index.intersection(backtest.time_index)
                if len(common_dates) == 0:
                    logger.warning(f"No overlapping dates between actual and {model_name} backtest")
                    metrics[model_name] = {metric: None for metric in ['MAPE', 'sMAPE', 'MAE', 'RMSE']}
                    continue
                
                actual_trimmed = actual.slice(intersect_from=common_dates[0], intersect_until=common_dates[-1])
                backtest_trimmed = backtest.slice(intersect_from=common_dates[0], intersect_until=common_dates[-1])
                
                logger.info(f"Trimmed actual length: {len(actual_trimmed)}, Trimmed backtest length: {len(backtest_trimmed)}")
                
                metrics[model_name] = {
                    'MAPE': safe_metric(mape, actual_trimmed, backtest_trimmed),
                    'sMAPE': safe_metric(smape, actual_trimmed, backtest_trimmed),
                    'MAE': safe_metric(mae, actual_trimmed, backtest_trimmed),
                    'RMSE': safe_metric(rmse, actual_trimmed, backtest_trimmed)
                }
                
                logger.info(f"Metrics for {model_name}: {metrics[model_name]}")
            else:
                logger.warning(f"Backtest for {model_name} is not a valid TimeSeries object")
                metrics[model_name] = {metric: None for metric in ['MAPE', 'sMAPE', 'MAE', 'RMSE']}
        else:
            logger.warning(f"No backtest results found for {model_name}")
            metrics[model_name] = {metric: None for metric in ['MAPE', 'sMAPE', 'MAE', 'RMSE']}
    
    return metrics
