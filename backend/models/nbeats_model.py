import numpy as np
import pytorch_lightning as pl
import streamlit as st
import torch
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import NBEATSModel
import traceback



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
    def __init__(self, input_chunk_length=24, output_chunk_length=12, n_epochs=50):
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.n_epochs = n_epochs
        self.model = None
        self.scaler = Scaler()

    def train(self, data: TimeSeries):
        st.text("Training N-BEATS model...")

        # Convert data to float32
        data_float32 = data.astype(np.float32)

        # Scale the data
        scaled_data = self.scaler.fit_transform(data_float32)

        # Determine the best accelerator and devices
        if torch.cuda.is_available():
            accelerator = "gpu"
            devices = 1
        elif torch.backends.mps.is_available():
            accelerator = "mps"
            devices = 1
        else:
            accelerator = "cpu"
            devices = "auto"

        # Create progress bar and status text
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Create and train the model
        self.model = NBEATSModel(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            n_epochs=self.n_epochs,
            random_state=42,
            pl_trainer_kwargs={
                "accelerator": accelerator,
                "devices": devices,
                "callbacks": [PrintEpochResults(progress_bar, status_text, self.n_epochs)],
                "enable_progress_bar": False,  # Disable default progress bar
            }
        )

        self.model.fit(scaled_data, verbose=False)  # Set verbose to False to avoid duplicate output
        st.text("N-BEATS model training completed")

    def predict(self, n: int) -> TimeSeries:
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")

        forecast = self.model.predict(n)
        return self.scaler.inverse_transform(forecast)

def train_nbeats_model(data: TimeSeries) -> NBEATSPredictor:
    model = NBEATSPredictor()
    model.train(data)
    return model

def make_nbeats_forecast(model: NBEATSPredictor, data: TimeSeries, forecast_horizon: int) -> TimeSeries:
    try:
        forecast = model.predict(forecast_horizon)
        return forecast
    except Exception as e:
        print(f"Error generating N-BEATS forecast: {type(e).__name__}: {str(e)}")
        raise
