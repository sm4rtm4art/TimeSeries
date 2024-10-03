import traceback
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.metrics import mae, mape, mse, rmse, smape


def calculate_metrics(test_data: TimeSeries, forecast: TimeSeries or list) -> dict:
    try:
        # Convert forecast to TimeSeries if it's a list
        if isinstance(forecast, list):
            if len(forecast) == 0:
                print("Warning: Empty forecast list")
                return {metric: None for metric in ["MAE", "MSE", "RMSE", "MAPE", "sMAPE"]}
            forecast = TimeSeries.from_series(pd.concat([f.pd_series() for f in forecast]))

        # Check if either test_data or forecast is empty
        if len(test_data) == 0 or len(forecast) == 0:
            print(f"Warning: Empty TimeSeries detected. Test data length: {len(test_data)}, Forecast length: {len(forecast)}")
            return {metric: None for metric in ["MAE", "MSE", "RMSE", "MAPE", "sMAPE"]}

        # Ensure the forecast and test data have the same time range
        common_start = max(test_data.start_time(), forecast.start_time())
        common_end = min(test_data.end_time(), forecast.end_time())

        test_data_trimmed = test_data.slice(common_start, common_end)
        forecast_trimmed = forecast.slice(common_start, common_end)

        # Check if trimmed data is empty
        if len(test_data_trimmed) == 0 or len(forecast_trimmed) == 0:
            print(f"Warning: Empty TimeSeries after trimming. Test data length: {len(test_data_trimmed)}, Forecast length: {len(forecast_trimmed)}")
            return {metric: None for metric in ["MAE", "MSE", "RMSE", "MAPE", "sMAPE"]}

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