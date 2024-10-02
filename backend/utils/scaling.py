from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
import numpy as np

def scale_data(data: TimeSeries) -> (TimeSeries, Scaler):
    scaler = Scaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data.astype(np.float32), scaler

def inverse_scale_forecast(forecast: TimeSeries, scaler: Scaler) -> TimeSeries:
    return scaler.inverse_transform(forecast).astype(np.float32)