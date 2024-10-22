"""
Chronos model
"""
import numpy as np
import pandas as pd
import streamlit as st
import torch
from chronos import ChronosPipeline
from darts import TimeSeries
from typing import Dict, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class ChronosPredictor:
    def __init__(self, model_size: str = "small"):
        self.model_size = model_size
        self.device_map = self._get_device_map()
        self.pipeline = self._initialize_pipeline()

    def _get_device_map(self):
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _initialize_pipeline(self):
        logger.info(f"Initializing Chronos model with size: {self.model_size}")
        model_name = f"amazon/chronos-t5-{self.model_size}"
        return ChronosPipeline.from_pretrained(
            model_name,
            device_map=self.device_map,
            torch_dtype=torch.float32 if self.device_map == "cpu" else torch.float16,
        )

    def train(self, data: TimeSeries) -> None:
        logger.info("Storing training data for Chronos model")
        self.training_data = data
        logger.info("Chronos model training completed")

    def predict(self, horizon: int, data: TimeSeries = None) -> TimeSeries:
        if self.pipeline is None:
            raise ValueError("Model has not been initialized. Reinitialize the ChronosPredictor.")

        if data is None:
            data = self.training_data

        if data is None:
            raise ValueError("No data provided for prediction.")

        logger.info(f"Predicting with Chronos model. Horizon: {horizon}")
        context = torch.tensor(data.values())
        forecast = self.pipeline.predict(context=context, prediction_length=horizon, num_samples=20)
        
        # Convert forecast to TimeSeries
        predictions = np.median(forecast[0].numpy(), axis=0)
        forecast_dates = pd.date_range(start=data.time_index[-1] + data.freq, periods=horizon, freq=data.freq)
        return TimeSeries.from_times_and_values(forecast_dates, predictions)

    def backtest(
        self,
        data: TimeSeries,
        forecast_horizon: int,
        start: Union[pd.Timestamp, float, int] = 0.5,
        retrain: bool = False,
        verbose: bool = False,
    ) -> Tuple[TimeSeries, Dict[str, float]]:
        logger.info(f"Starting Chronos backtest. Data length: {len(data)}, Forecast horizon: {forecast_horizon}, Start: {start}")
        
        if isinstance(start, float):
            start = int(len(data) * start)
        
        start_timestamp = data.time_index[start]
        
        historical_forecasts = self.historical_forecasts(
            data, 
            start=start_timestamp, 
            forecast_horizon=forecast_horizon
        )
        
        # Ensure the historical forecasts match the test data length
        actual_data = data.slice(start_timestamp, data.end_time())
        if len(historical_forecasts) != len(actual_data):
            logger.warning(f"Adjusting historical forecasts length from {len(historical_forecasts)} to {len(actual_data)}")
            historical_forecasts = historical_forecasts.slice(actual_data.start_time(), actual_data.end_time())
        
        # Calculate metrics
        from backend.utils.metrics import calculate_metrics
        metrics = calculate_metrics(actual_data, historical_forecasts)
        
        logger.info(f"Backtest completed. Forecast length: {len(historical_forecasts)}")
        logger.info(f"Backtest metrics: {metrics}")
        
        return historical_forecasts, metrics

    def historical_forecasts(
        self,
        series: TimeSeries,
        start: pd.Timestamp,
        forecast_horizon: int,
        stride: int = 1,
        retrain: bool = False,
        verbose: bool = False,
    ) -> TimeSeries:
        logger.info(f"Generating historical forecasts for Chronos. Start: {start}, Horizon: {forecast_horizon}")
        historical_forecasts = []
        for i in range(0, len(series) - forecast_horizon + 1, stride):
            if series.time_index[i] >= start:
                forecast = self.predict(forecast_horizon, series[:i])
                historical_forecasts.append(forecast)

        if not historical_forecasts:
            raise ValueError("No historical forecasts generated. Check your start date and forecast horizon.")

        return TimeSeries.from_series(pd.concat([f.pd_series() for f in historical_forecasts]))

def train_chronos_model(train_data: TimeSeries, model_size: str = "small") -> ChronosPredictor:
    """
    Train a Chronos model using the ChronosPredictor class.
    
    Args:
    train_data (TimeSeries): The training data as a Darts TimeSeries object.
    model_size (str): Size of the Chronos model. Options: "tiny", "small", "medium", "large".
    
    Returns:
    ChronosPredictor: A trained ChronosPredictor instance.
    """
    st.text(f"Training Chronos model (size: {model_size})...")
    model = ChronosPredictor(model_size=model_size)
    try:
        model.train(train_data)
        st.text("Chronos model training completed")
    except Exception as e:
        st.error(f"Error during Chronos model training: {str(e)}")
        raise
    return model


def make_chronos_forecast(model: ChronosPredictor, data: TimeSeries, forecast_horizon: int) -> TimeSeries:
    return model.predict(forecast_horizon, data)
