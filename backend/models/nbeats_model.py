import traceback

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import streamlit as st
import torch
from darts import TimeSeries
from darts.models import NBEATSModel
from pytorch_lightning.callbacks import EarlyStopping

from backend.utils.scaling import scale_data


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

    def predict(self, data: TimeSeries, horizon: int) -> TimeSeries:
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
        scaled_data, _ = scale_data(data.astype(np.float32))
        forecast = self.model.predict(n=horizon, series=scaled_data)
        return self.scaler.inverse_transform(forecast)

    def historical_forecast(self, data: TimeSeries, start: int, forecast_horizon: int) -> TimeSeries:
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
        scaled_data, _ = scale_data(data.astype(np.float32))
        historical_forecast = self.model.historical_forecasts(
            scaled_data,
            start=start,
            forecast_horizon=forecast_horizon,
            retrain=False,
            verbose=True
        )
        return self.scaler.inverse_transform(historical_forecast)

    def backtest(self, data: TimeSeries, start: int, forecast_horizon: int) -> TimeSeries:
            if self.model is None:
                raise ValueError("Model has not been trained. Call train() first.")
            scaled_data, _ = scale_data(data.astype(np.float32))
            backtest_forecast = self.model.historical_forecasts(
                scaled_data,
                start=start,
                forecast_horizon=forecast_horizon,
                stride=1,
                retrain=False,
                verbose=True
            )
            return self.scaler.inverse_transform(backtest_forecast)
