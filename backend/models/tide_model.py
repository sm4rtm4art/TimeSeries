import logging
from typing import Optional, Tuple
import traceback


import numpy as np
import pytorch_lightning as pl
import streamlit as st
import torch
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TiDEModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_best_accelerator():
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



def train_tide_model(data: TimeSeries) -> Tuple[TiDEModel, Scaler]:
    print("Training TiDE model...")
    st.text("Training TiDE model...")
    
    # Convert data to float32
    data_float32 = data.astype(np.float32)

    # Scale the data
    scaler = Scaler()
    scaled_data = scaler.fit_transform(data_float32)

    accelerator = get_best_accelerator()
    print(f"Using accelerator: {accelerator}")

    progress_bar = st.progress(0)
    status_text = st.empty()

    n_epochs = 100  # You can adjust this value as needed

    model = TiDEModel(
        input_chunk_length=24,
        output_chunk_length=12,
        decoder_output_dim=64,
        temporal_width_past=16,
        temporal_width_future=16,
        temporal_decoder_hidden=64,
        use_layer_norm=True,
        use_reversible_instance_norm=True,
        n_epochs=n_epochs,
        batch_size=32,
        optimizer_kwargs={'lr': 1e-3},
        random_state=42,
        pl_trainer_kwargs={
            "accelerator": accelerator,
            "precision": "32-true",
            "enable_model_summary": False,
            "callbacks": [PrintEpochResults(progress_bar, status_text, n_epochs)],
            "log_every_n_steps": 1,
        },
        # Remove the dtype parameter
    )

    try:
        model.fit(scaled_data, verbose=False)
        print("TiDE model training completed")
        st.text("TiDE model training completed")
    except Exception as e:
        error_msg = f"Error during TiDE model training: {type(e).__name__}: {str(e)}"
        print(error_msg)
        print("Traceback:")
        traceback.print_exc()
        st.error(error_msg)
        raise

    return model, scaler


def make_tide_forecast(model, scaler, data: TimeSeries, forecast_horizon: int) -> TimeSeries:
    try:
        print(f"Starting TiDE forecast generation. Input data length: {len(data)}, Forecast horizon: {forecast_horizon}")
        
        # Ensure the input data has at least input_chunk_length points
        input_chunk_length = model.input_chunk_length
        print(f"Model input_chunk_length: {input_chunk_length}")
        
        if len(data) < input_chunk_length:
            padding_length = input_chunk_length - len(data)
            padding = data.slice(-padding_length)  # Use the last padding_length points for padding
            padded_data = padding.concatenate(data)
            print(f"Data padded. New length: {len(padded_data)}")
        else:
            padded_data = data
            print("No padding needed")

        # Use the last available data points as the starting point for the forecast
        last_data_points = padded_data[-input_chunk_length:]
        print(f"Using last {len(last_data_points)} points for forecast")
        
        # Convert to float32
        last_data_points = last_data_points.astype(np.float32)
        
        # Scale the data
        scaled_data = scaler.transform(last_data_points)
        print(f"Data scaled. Scaled data length: {len(scaled_data)}, Start time: {scaled_data.start_time()}, End time: {scaled_data.end_time()}")
        
        # Generate forecast
        scaled_forecast = model.predict(n=forecast_horizon, series=scaled_data)
        print(f"Raw forecast generated. Length: {len(scaled_forecast)}")
        
        # Inverse transform the forecast
        forecast = scaler.inverse_transform(scaled_forecast)
        print(f"Forecast inverse transformed. Length: {len(forecast)}")
        
        # Ensure the forecast has the correct time index
        start_date = data.end_time() + data.freq
        forecast = forecast.slice(start_date, start_date + (forecast_horizon - 1) * data.freq)
        
        print(f"Final TiDE forecast: Length = {len(forecast)}, Start time = {forecast.start_time()}, End time = {forecast.end_time()}")
        
        return forecast
    except Exception as e:
        print(f"Error generating TiDE forecast: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        raise
