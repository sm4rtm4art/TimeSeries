import traceback

import numpy as np
import pytorch_lightning as pl
import streamlit as st
import torch
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TiDEModel


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

        # Convert data to float32
        data_float32 = data.astype(np.float32)

        # Scale the data
        scaled_data = self.scaler.fit_transform(data_float32)

        # Determine the best accelerator
        if torch.cuda.is_available():
            accelerator = "gpu"
        elif torch.backends.mps.is_available():
            accelerator = "mps"
        else:
            accelerator = "cpu"

        # Create progress bar and status text
        progress_bar = st.progress(0)
        status_text = st.empty()

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
                "accelerator": accelerator,
                "precision": "32-true",
                "enable_model_summary": False,
                "callbacks": [PrintEpochResults(progress_bar, status_text, self.n_epochs)],
                "log_every_n_steps": 1,
            }
        )

        self.model.fit(scaled_data, verbose=True)
        st.text("TiDE model training completed")

    def predict(self, n: int) -> TimeSeries:
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")

        forecast = self.model.predict(n)
        return self.scaler.inverse_transform(forecast)

    def backtest(self, data: TimeSeries, start: int, forecast_horizon: int) -> TimeSeries:
        # Implement TiDE backtesting logic here
        pass

    def predict(self, horizon: int) -> TimeSeries:
        # Implement TiDE prediction logic here
        pass

def train_tide_model(data: TimeSeries) -> TiDEPredictor:
    model = TiDEPredictor()
    model.train(data)
    return model

def make_tide_forecast(model: TiDEPredictor, data: TimeSeries, forecast_horizon: int) -> TimeSeries:
    try:
        print(f"Starting TiDE forecast generation. Input data length: {len(data)}, Forecast horizon: {forecast_horizon}")

        # Generate forecast
        forecast = model.predict(forecast_horizon)
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
