
import numpy as np
import streamlit as st
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import NBEATSModel


class NBEATSPredictor:
    def __init__(self, input_chunk_length: int = 24, output_chunk_length: int = 12, n_epochs: int = 50):
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.n_epochs = n_epochs
        self.model = None
        self.scaler = Scaler()

    def train(self, data: TimeSeries) -> None:
        try:
            # Convert data to float32
            data_float32 = data.astype(np.float32)
            series_scaled = self.scaler.fit_transform(data_float32)
            self.model = NBEATSModel(
                input_chunk_length=self.input_chunk_length,
                output_chunk_length=self.output_chunk_length,
                n_epochs=self.n_epochs,
                force_reset=True,
                model_name="nbeats_model",
                pl_trainer_kwargs={"accelerator": "gpu", "precision": "32-true"}  # Changed to CPU
            )
            self.model.fit(series_scaled)
            st.success("N-BEATS model trained successfully!")
        except Exception as e:
            st.error(f"Error training N-BEATS model: {str(e)}")
            raise

    def predict(self, periods: int) -> TimeSeries:
        try:
            if self.model is None:
                raise ValueError("Model has not been trained. Call train() first.")

            forecast = self.model.predict(periods)
            return self.scaler.inverse_transform(forecast)
        except Exception as e:
            st.error(f"Error predicting with N-BEATS model: {str(e)}")
            raise

def train_nbeats_model(data: TimeSeries) -> NBEATSPredictor:
    model = NBEATSPredictor()
    model.train(data)
    return model

def make_nbeats_forecast(model: NBEATSPredictor, horizon: int) -> TimeSeries:
    return model.predict(horizon)
