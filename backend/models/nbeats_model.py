import numpy as np
import pytorch_lightning as pl
import streamlit as st
import torch
import pandas as pd

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import NBEATSModel
import traceback
from pytorch_lightning.callbacks import EarlyStopping
from backend.utils.tensor_utils import ensure_float32, is_mps_available
from backend.utils.scaling import scale_data, inverse_scale_forecast

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
        self.status_text.text(f"Training N-BEATS model: Epoch {current_epoch + 1}/{self.total_epochs}, Loss: {loss:.4f}")

class NBEATSPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.n_epochs = 100

    def train(self, data: TimeSeries):
        st.text("Training N-BEATS model...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Convert data to float32
        data_float32 = data.astype(np.float32)
        scaled_data, self.scaler = scale_data(data_float32)

        # Create callbacks
        early_stopping = EarlyStopping(monitor="train_loss", patience=5, mode="min")
        print_epoch_results = PrintEpochResults(progress_bar, status_text, self.n_epochs)

        # Create the model
        self.model = NBEATSModel(
            input_chunk_length=24,
            output_chunk_length=12,
            generic_architecture=True,
            num_stacks=10,
            num_blocks=1,
            num_layers=4,
            layer_widths=512,
            n_epochs=self.n_epochs,
            nr_epochs_val_period=1,
            batch_size=800,
            model_name="nbeats_run",
            pl_trainer_kwargs={
                "accelerator": determine_accelerator(),
                "precision": "32-true",
                "enable_model_summary": False,
                "callbacks": [early_stopping, print_epoch_results],
                "log_every_n_steps": 1,
                "enable_progress_bar": False,
            }
        )

        # Train the model
        self.model.fit(scaled_data, verbose=True)
        st.text("N-BEATS model training completed")

    def predict(self, horizon: int) -> TimeSeries:
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
        forecast = self.model.predict(n=horizon)
        return self.scaler.inverse_transform(forecast.astype(np.float32))

    def historical_forecast(self, data: TimeSeries, start: int, forecast_horizon: int) -> TimeSeries:
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
        historical_forecast = self.model.historical_forecasts(
            data.astype(np.float32),
            start=start,
            forecast_horizon=forecast_horizon,
            retrain=False,
            verbose=True
        )
        return self.scaler.inverse_transform(historical_forecast).astype(np.float32)

    def backtest(self, data: TimeSeries, start: int, forecast_horizon: int) -> TimeSeries:
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
        data_float32 = data.astype(np.float32)
        scaled_data, _ = scale_data(data_float32)
        
        backtest_forecasts = []
        for i in range(start, len(data)):
            forecast = self.model.historical_forecasts(
                scaled_data,
                start=i,
                forecast_horizon=1,
                retrain=False,
                verbose=False
            )
            backtest_forecasts.append(forecast)
        
        combined_forecast = TimeSeries.from_series(pd.concat([f.pd_series() for f in backtest_forecasts]))
        return self.scaler.inverse_transform(combined_forecast.astype(np.float32))


def make_nbeats_forecast(model: NBEATSPredictor, data: TimeSeries, forecast_horizon: int) -> TimeSeries:
    try:
        print(f"Generating N-BEATS forecast for horizon: {forecast_horizon}")
        print(f"Input data: Length = {len(data)}, Start = {data.start_time()}, End = {data.end_time()}")
        
        # Convert data to float32 and scale
        data_float32 = data.astype(np.float32)
        scaled_data, _ = scale_data(data_float32)
        
        # Generate forecast
        forecast = model.model.predict(n=forecast_horizon, series=scaled_data)
        print(f"Forecast generated. Length: {len(forecast)}, Start: {forecast.start_time()}, End: {forecast.end_time()}")
        
        # Inverse transform the forecast
        inverse_forecast = model.scaler.inverse_transform(forecast)
        
        # Clip values to be within a reasonable range
        historical_min = data.min().values()[0]
        historical_max = data.max().values()[0]
        padding = (historical_max - historical_min) * 0.2  # Allow 20% outside historical range
        clipped_forecast = inverse_forecast.clip(lower=historical_min - padding, upper=historical_max + padding)
        
        print(f"Clipped forecast: Min = {clipped_forecast.min().values()[0]}, Max = {clipped_forecast.max().values()[0]}")
        
        return clipped_forecast
    except Exception as e:
        print(f"Error generating N-BEATS forecast: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        raise
