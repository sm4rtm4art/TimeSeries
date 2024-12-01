from typing import Dict, Any, Optional
from darts import TimeSeries
import streamlit as st
import logging
import traceback
import numpy as np

from backend.core.interfaces.base_model import TimeSeriesPredictor
from backend.core.model_factory import ModelFactory
from backend.core.exceptions import ModelTrainingError

logger = logging.getLogger(__name__)

class ModelTrainer:
    @staticmethod
    def train_model(
        model: TimeSeriesPredictor,
        train_data: TimeSeries,
    ) -> TimeSeriesPredictor:
        """Train a single model"""
        try:
            # Convert data to float32 before training
            train_data_float32 = train_data.astype(np.float32)
            logger.info(f"Converting data to float32. Original dtype: {train_data.dtype}, New dtype: {train_data_float32.dtype}")
            
            model.train(train_data_float32)
            return model
        except Exception as e:
            logger.error(f"Error training {model.model_name}: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    @staticmethod
    def train_models(
        train_data: TimeSeries,
        model_choice: str = "All Models",
        model_size: str = "small"
    ) -> Dict[str, TimeSeriesPredictor]:
        """Train multiple models"""
        logger.info(f"Creating models with choice: {model_choice}, size: {model_size}")
        
        trained_models = {}
        models_to_train = ModelFactory.create_models(model_choice, model_size)
        
        logger.info(f"Models created: {list(models_to_train.keys()) if models_to_train else 'None'}")
        
        if not models_to_train:
            raise ModelTrainingError(f"No models created for choice: {model_choice}. Available choices should be: Prophet, NBEATS, TiDE, TSMixer")
        
        for model_name, model in models_to_train.items():
            try:
                with st.spinner(f"Training {model_name}..."):
                    trained_models[model_name] = ModelTrainer.train_model(model, train_data)
                    st.success(f"{model_name} trained successfully!")
            except Exception as e:
                error_msg = f"Error training {model_name}: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                st.error(error_msg)
                raise ModelTrainingError(error_msg) from e
                
        return trained_models if trained_models else None