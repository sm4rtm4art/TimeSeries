from typing import Dict, Optional
from darts import TimeSeries
import pytorch_lightning as pl
import torch
import streamlit as st
from backend.core.interfaces.model import TimeSeriesPredictor
import logging

logger = logging.getLogger(__name__)

class ModelTrainingService:
    @staticmethod
    def determine_accelerator() -> str:
        """
        Determine the available accelerator for PyTorch.
        For Apple Silicon Macs, prefer MPS over CPU.
        """
        if torch.cuda.is_available():
            return "gpu"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    class PrintEpochResults(pl.Callback):
        """Callback to print epoch results during model training."""
        
        def __init__(self, progress_bar, status_text, total_epochs: int, model_name: str = ""):
            super().__init__()
            self.progress_bar = progress_bar
            self.status_text = status_text
            self.total_epochs = total_epochs
            self.model_name = model_name

        def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
            current_epoch = trainer.current_epoch
            loss = trainer.callback_metrics["train_loss"].item()
            progress = (current_epoch + 1) / self.total_epochs
            self.progress_bar.progress(progress)
            model_prefix = f"{self.model_name} " if self.model_name else ""
            self.status_text.text(
                f"Training {model_prefix}model: Epoch {current_epoch + 1}/{self.total_epochs}, Loss: {loss:.4f}"
            )

    def __init__(self, model_registry: Dict[str, TimeSeriesPredictor]):
        self.model_registry = model_registry

    def train_model(
        self, 
        data: TimeSeries, 
        model_choice: str, 
        model_size: str,
        progress_bar: Optional[st.progress] = None,
        status_text: Optional[st.empty] = None
    ) -> TimeSeriesPredictor:
        """Train a specific model with progress tracking."""
        if model_choice not in self.model_registry:
            raise ValueError(f"Unknown model: {model_choice}")
        
        try:
            logger.info(f"Training {model_choice} model")
            model = self.model_registry[model_choice]()
            
            # Set up progress tracking if provided
            if progress_bar and status_text:
                model.setup_progress_tracking(progress_bar, status_text)
                
            model.train(data)
            logger.info(f"{model_choice} model trained successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error training {model_choice} model: {str(e)}")
            raise 