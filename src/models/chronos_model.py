import torch
import numpy as np
from chronos import ChronosPipeline
from darts import TimeSeries

class ChronosPredictor:
    def __init__(self, model_size: str = "small"):
        self.model_size = model_size
        self.pipeline = None
        self.device_map = "mps"  # Use "cpu" for CPU inference and "mps" for Apple Silicon

    def train(self, data: TimeSeries) -> None:
        model_name = f"amazon/chronos-t5-{self.model_size}"
        self.pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=self.device_map,
            torch_dtype=torch.bfloat16,
        )

    def predict(self, periods: int) -> TimeSeries:
        if self.pipeline is None:
            raise ValueError("Model has not been trained. Call train() first.")

        # Convert TimeSeries to tensor
        context = torch.tensor(data.values())
        forecast = self.pipeline.predict(context, periods)
        predictions = np.quantile(forecast.numpy(), 0.5, axis=1)

        # Convert predictions back to TimeSeries
        return TimeSeries.from_values(predictions)

def make_chronos_forecast(model: ChronosPredictor, data: TimeSeries, forecast_horizon: int) -> TimeSeries:
    return model.predict(forecast_horizon)