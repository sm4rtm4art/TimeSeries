import traceback
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import streamlit as st
import torch
from darts import TimeSeries
from darts.models import NBEATSModel
from pytorch_lightning.callbacks import EarlyStopping
from darts.dataprocessing.transformers import Scaler



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
    def __init__(self, input_chunk_length=24, output_chunk_length=12):
        self.n_epochs = 100
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length

        # Create progress bar and status text
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Create callbacks
        early_stopping = EarlyStopping(monitor="train_loss", patience=10, min_delta=0.000001, mode="min")
        print_epoch_results = PrintEpochResults(progress_bar, status_text, self.n_epochs)

        self.model = NBEATSModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            generic_architecture=True,
            num_stacks=10,
            num_blocks=1,
            num_layers=4,
            layer_widths=512,
            expansion_coefficient_dim=5,
            trend_polynomial_degree=2,
            batch_size=800,
            n_epochs=self.n_epochs,
            nr_epochs_val_period=1,
            dropout=0.1,
            model_name="nbeats_interpretable_run",
            pl_trainer_kwargs={
                "accelerator": determine_accelerator(),
                "precision": "32-true",
                "enable_model_summary": True,
                "callbacks": [#early_stopping,
                               print_epoch_results],
                "log_every_n_steps": 1,
                "enable_progress_bar": False
            }
        )
        self.scaler = None

    def train(self, data: TimeSeries):
        st.text("Training N-BEATS model...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Convert data to float32
        data_float32 = data.astype(np.float32)
        scaled_data, self.scaler = scale_data(data_float32)

        # Create callbacks
        early_stopping = EarlyStopping(monitor="train_loss", patience=10, mode="min")
        print_epoch_results = PrintEpochResults(progress_bar, status_text, self.n_epochs)

        # Train the model
        self.model.fit(scaled_data, verbose=True)
        st.text("N-BEATS model training completed")

    def predict(self, horizon: int, data: TimeSeries = None) -> TimeSeries:
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
        
        if data is None:
            data = self.model.training_series
        
        # Use the last input_chunk_length points from the provided data
        data = data[-self.input_chunk_length:]
    
        scaled_data, _ = scale_data(data.astype(np.float32))
        forecast = self.model.predict(n=horizon, series=scaled_data)
        return self.scaler.inverse_transform(forecast)

    def historical_forecasts(
        self,
        series: TimeSeries,
        start: Union[pd.Timestamp, int, float],
        forecast_horizon: int,
        stride: int = 1,
        retrain: bool = False,
        verbose: bool = False,
    ) -> TimeSeries:
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
        print(f"Historical forecast requested from {start} for {forecast_horizon} steps")
        print(f"Series range: {series.start_time()} to {series.end_time()}")

        # Convert start to pd.Timestamp if it's not already
        if isinstance(start, int):
            start = series.time_index[start]
        elif isinstance(start, float):
            start_index = int(len(series) * start)
            start = series.time_index[start_index]
        
        if not isinstance(start, pd.Timestamp):
            raise ValueError("start must be a pd.Timestamp, int index, or float proportion")

        # Ensure start is within the series timeframe
        if start >= series.end_time():
            raise ValueError(f"Start time {start} is at or after the last timestamp {series.end_time()} of the series.")

        # Adjust forecast horizon if it goes beyond the end of the series
        if start + pd.Timedelta(days=forecast_horizon) > series.end_time():
            forecast_horizon = (series.end_time() - start).days
            print(f"Adjusted forecast horizon to {forecast_horizon} to fit within available data")

        scaled_series, _ = scale_data(series.astype(np.float32))

        try:
            historical_forecast = self.model.historical_forecasts(
                scaled_series,
                start=start,
                forecast_horizon=forecast_horizon,
                stride=stride,
                retrain=retrain,
                verbose=verbose,
                overlap_end=True
            )
            print(f"Historical forecast generated successfully. Length: {len(historical_forecast)}")
            return self.scaler.inverse_transform(historical_forecast)
        except Exception as e:
            print("Error in historical forecasts:")
            print(traceback.format_exc())
            return None

    def backtest(
        self,
        data: TimeSeries,
        forecast_horizon: int,
        start: Union[float, int]
    ) -> Tuple[TimeSeries, Dict[str, float]]:
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")

        # Convert start to pd.Timestamp
        if isinstance(start, int):
            start_timestamp = data.time_index[start]
        elif isinstance(start, float):
            start_index = int(len(data) * start)
            start_timestamp = data.time_index[start_index]
        else:
            raise ValueError("start must be a float between 0 and 1 or an integer index.")

        # Perform backtesting
        backtest_series = data.slice(start_timestamp, data.end_time())
        historical_forecasts = self.historical_forecasts(
            series=data,
            start=start_timestamp,
            forecast_horizon=forecast_horizon,
            stride=1,
            retrain=False,
            verbose=False
        )
        
        # Calculate metrics
        metrics = self.evaluate(backtest_series, historical_forecasts)
        
        return historical_forecasts, metrics

    def evaluate(self, actual: TimeSeries, predicted: TimeSeries) -> Dict[str, float]:
        from darts.metrics import mape, mse, rmse
        return {
            "MAPE": mape(actual, predicted),
            "MSE": mse(actual, predicted),
            "RMSE": rmse(actual, predicted)
        }