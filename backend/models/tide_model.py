import traceback
from typing import Dict, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import streamlit as st
import torch
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TiDEModel
from darts.metrics import mape, mse, rmse

from pytorch_lightning.callbacks import EarlyStopping

def determine_accelerator():
    if torch.cuda.is_available():
        return "gpu"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

class PrintEpochResults(pl.Callback):
    def __init__(self, progress_bar, status_text, total_epochs):
        super().__init__()
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.total_epochs = total_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        loss = trainer.callback_metrics['train_loss'].item()
        progress = (current_epoch + 1) / self.total_epochs
        self.progress_bar.progress(progress)
        self.status_text.text(f"Training TiDE model: Epoch {current_epoch + 1}/{self.total_epochs}, Loss: {loss:.4f}")

class TiDEPredictor:
    def __init__(self, input_chunk_length=24, output_chunk_length=12, n_epochs=100):
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.n_epochs = n_epochs
        self.model = None
        self.scaler = Scaler()

    def train(self, data: TimeSeries):
        st.text("Training TiDE model...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Convert data to float32
        data_float32 = data.astype(np.float32)

        # Scale the data
        scaled_data = self.scaler.fit_transform(data_float32)

        early_stopping = EarlyStopping(monitor="train_loss", patience=10, min_delta=0.000001, mode="min")
        print_epoch_results = PrintEpochResults(progress_bar, status_text, self.n_epochs)

        # Create and train the model
        self.model = TiDEModel(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            decoder_output_dim=64,
            temporal_width_past=16,
            temporal_width_future=16,
            temporal_decoder_hidden=64,
            use_layer_norm=True,
            use_reversible_instance_norm=True,
            n_epochs=self.n_epochs,
            batch_size=32,
            optimizer_kwargs={'lr': 1e-3},
            random_state=42,
            pl_trainer_kwargs={
                "accelerator": determine_accelerator(),
                "precision": "32-true",
                "enable_model_summary": True,
                "callbacks": [early_stopping, print_epoch_results],
                "log_every_n_steps": 1,
                "enable_progress_bar": False
                }
        )

        self.model.fit(scaled_data, verbose=True)
        st.text("TiDE model training completed")

    def predict(self, horizon: int, data: TimeSeries = None) -> TimeSeries:
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
        
        if data is not None:
            scaled_data = self.scaler.transform(data.astype(np.float32))
        else:
            scaled_data = self.scaler.transform(self.model.training_series.astype(np.float32))
        
        forecast = self.model.predict(n=horizon, series=scaled_data)
        return self.scaler.inverse_transform(forecast)

    def historical_forecasts(self, series, start, forecast_horizon, stride=1, retrain=False, verbose=False):
        scaled_series = self.scaler.transform(series.astype(np.float32))
        historical_forecasts = self.model.historical_forecasts(
            series=scaled_series,
            start=start,
            forecast_horizon=forecast_horizon,
            stride=stride,
            retrain=retrain,
            verbose=verbose,
            last_points_only=False
        )
        return self.scaler.inverse_transform(historical_forecasts)

    def backtest(
        self,
        data: TimeSeries,
        forecast_horizon: int,
        start: Union[float, int]
    ) -> Tuple[TimeSeries, Dict[str, float]]:
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")

        print(f"Starting backtesting for TiDE model. Forecast horizon: {forecast_horizon}")

        # Convert start to pd.Timestamp
        if isinstance(start, int):
            start_timestamp = data.time_index[start]
        elif isinstance(start, float):
            start_index = int(len(data) * start)
            start_timestamp = data.time_index[start_index]
        else:
            raise ValueError("start must be a float between 0 and 1 or an integer index.")

        print(f"Backtest start timestamp: {start_timestamp}")

        # Perform backtesting using historical_forecasts
        historical_forecasts = self.historical_forecasts(
            series=data,
            start=start_timestamp,
            forecast_horizon=forecast_horizon,
            stride=1,
            retrain=False,
            verbose=True
        )
        
        # Convert list of forecasts to a single TimeSeries if necessary
        if isinstance(historical_forecasts, list):
            print(f"Converting list of {len(historical_forecasts)} forecasts to a single TimeSeries")
            historical_forecasts = TimeSeries.from_series(pd.concat([f.pd_series() for f in historical_forecasts]))
        
        print(f"Historical forecasts generated. Length: {len(historical_forecasts)}, Start: {historical_forecasts.start_time()}, End: {historical_forecasts.end_time()}")

        # Prepare actual data for comparison
        actual_data = data.slice(start_timestamp, data.end_time())

        print(f"Actual data prepared. Length: {len(actual_data)}, Start: {actual_data.start_time()}, End: {actual_data.end_time()}")

        # Calculate metrics
        metrics = self.evaluate(actual_data, historical_forecasts)
        
        print(f"Metrics calculated: {metrics}")

        return historical_forecasts, metrics

    def evaluate(self, actual: TimeSeries, predicted: TimeSeries) -> Dict[str, float]:
        # Ensure the time ranges match
        actual_trimmed, predicted_trimmed = actual.slice_intersect(predicted), predicted.slice_intersect(actual)
        
        return {
            "MAPE": mape(actual_trimmed, predicted_trimmed),
            "MSE": mse(actual_trimmed, predicted_trimmed),
            "RMSE": rmse(actual_trimmed, predicted_trimmed)
        }

def train_tide_model(data: TimeSeries) -> TiDEPredictor:
    model = TiDEPredictor()
    model.train(data)
    return model

def make_tide_forecast(model: TiDEPredictor, data: TimeSeries, forecast_horizon: int) -> TimeSeries:
    try:
        print(f"Starting TiDE forecast generation. Input data length: {len(data)}, Forecast horizon: {forecast_horizon}")

        # Generate forecast
        forecast = model.predict(forecast_horizon, data.astype(np.float32))
        print(f"Forecast generated. Length: {len(forecast)}")

        # Ensure the forecast has the correct time index
        start_date = data.end_time() + data.freq
        forecast = forecast.slice(start_date, start_date + (forecast_horizon - 1) * data.freq)

        print(f"Final TiDE forecast: Length = {len(forecast)}, Start time = {forecast.start_time()}, End time = {forecast.end_time()}")

        return forecast
    except Exception as e:
        print(f"Error generating TiDE forecast: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        raise