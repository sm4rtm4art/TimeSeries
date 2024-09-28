import numpy as np
from darts import TimeSeries
from darts.metrics import mae, mse, rmse


def split_data(data: TimeSeries, test_duration: int = 24) -> tuple[TimeSeries, TimeSeries]:
    """
    Split the data into train and test sets.

    :param data: The full time series data
    :param test_duration: Number of periods for the test set (default is 24 for 2 years of monthly data)
    :return: A tuple of (train_data, test_data)
    """
    total_length = len(data)
    train_length = total_length - test_duration
    train_data = data[:train_length]
    test_data = data[train_length:]
    return train_data, test_data


def calculate_metrics(actual: TimeSeries, forecast: TimeSeries) -> dict:
    """
    Calculate forecast accuracy metrics.
 
    :param actual: The actual time series data
    :param forecast: The forecasted time series data
    :return: A dictionary of metric names and values
    """
    if len(actual) == 0 or len(forecast) == 0:
        raise ValueError("Actual or forecast TimeSeries is empty")

    if len(actual) != len(forecast):
        raise ValueError("Actual and forecast TimeSeries must have the same length")

    actual_values = actual.values()
    forecast_values = forecast.values()

    if np.isnan(actual_values).all() or np.isnan(forecast_values).all():
        raise ValueError("Actual or forecast TimeSeries contains only NaN values")

    metrics = {}
    metric_functions = {
        'MAPE': mape,
        'RMSE': rmse,
        'MAE': mae
    }

    for metric_name, metric_func in metric_functions.items():
        try:
            metrics[metric_name] = metric_func(actual, forecast)
        except (ValueError, ZeroDivisionError):
            metrics[metric_name] = np.nan

    return metrics


def calculate_metrics(test_data: TimeSeries, forecast: TimeSeries) -> dict:
    return {
        "MAE": mae(test_data, forecast),
        "MSE": mse(test_data, forecast),
        "RMSE": rmse(test_data, forecast)
    }
