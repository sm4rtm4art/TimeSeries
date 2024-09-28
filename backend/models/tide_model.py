import logging
from typing import Optional, Tuple

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
    """
    Train a TiDE model using the provided time series data.

    :param data: Input time series data
    :return: Tuple of (Trained TiDEModel instance, Scaler)
    """ 

    logger.info("Training TiDE model...")
    st.text("Training TiDE model...")
    
    # Convert data to float32
    data_float32 = data.astype(np.float32)

    # Scale the data
    scaler = Scaler()
    scaled_data = scaler.fit_transform(data_float32)

    accelerator = get_best_accelerator()
    logger.info(f"Using accelerator: {accelerator}")

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
        force_reset=True,
        model_name="tide_model",
    )

    try:
        model.fit(scaled_data, verbose=False)
        logger.info("TiDE model training completed")
        st.text("TiDE model training completed")
    except Exception as e:
        logger.error(f"Error during TiDE model training: {str(e)}")
        st.error(f"Error during TiDE model training: {str(e)}")
        raise

    return model, scaler


def make_tide_forecast(model, data: TimeSeries, forecast_horizon: int) -> Tuple[Optional[TimeSeries], Optional[str]]:
    try:
        forecast = model.predict(n=forecast_horizon)
        return forecast, None
    except Exception as e:
        return None, str(e)

