"""
Chronos model
"""
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline
from darts import TimeSeries


class ChronosPredictor:
    def __init__(self, model_size: str = "small"):
        self.model_size = model_size
        self.pipeline = None
        self.device_map = self._get_device_map()

    def _get_device_map(self):
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def train(self, data: TimeSeries) -> None:
        model_name = f"amazon/chronos-t5-{self.model_size}"
        self.pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=self.device_map,
            torch_dtype=torch.float32 if self.device_map == "cpu" else torch.float16,
        )

    def predict(self, data: TimeSeries, periods: int) -> TimeSeries:
        if self.pipeline is None:
            raise ValueError("Model has not been trained. Call train() first.")

        # Convert TimeSeries to tensor
        context = torch.tensor(data.values())
        forecast = self.pipeline.predict(context, periods)
        predictions = np.quantile(forecast.numpy(), 0.5, axis=1)

        # Convert predictions back to TimeSeries
        return TimeSeries.from_values(predictions)

    def set_model_size(self, model_size: str) -> None:
        """
        Set the model size and reset the pipeline.
        :param model_size: Size of the model ("tiny", "small", "medium", or "large")
        """
        self.model_size = model_size
        self.pipeline = None  # Reset the pipeline so it will be reloaded with the new size on next train() call


def make_chronos_forecast(model: ChronosPredictor, data: TimeSeries, forecast_horizon: int) -> TimeSeries:
    # Convert TimeSeries to tensor
    context = torch.tensor(data.values())
    forecast = model.pipeline.predict(context, forecast_horizon)
    predictions = np.quantile(forecast.numpy(), 0.5, axis=1)
    # Ensure predictions match the forecast horizon
    predictions = predictions[:forecast_horizon]
    # Create a new TimeSeries for the forecast
    forecast_dates = pd.date_range(start=data.end_time() + data.freq, periods=forecast_horizon, freq=data.freq)

    # Ensure the number of dates matches the number of predictions
    if len(forecast_dates) != len(predictions):
        raise ValueError(f"Mismatch between number of forecast dates ({len(forecast_dates)}) and predictions ({len(predictions)})")

    return TimeSeries.from_times_and_values(times=forecast_dates, values=predictions)
