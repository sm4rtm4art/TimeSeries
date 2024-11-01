from typing import Dict, Any
from darts import TimeSeries
import streamlit as st
import logging
import traceback

# Use absolute imports when being called from app
from backend.models.base_model import BasePredictor
from backend.core.model_factory import ModelFactory

logger = logging.getLogger(__name__)

class ModelTrainer:
    @staticmethod
    def train_model(
        model: BasePredictor,
        train_data: TimeSeries,
    ) -> BasePredictor:
        """Train a single model"""
        try:
            model.train(train_data)
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
    ) -> Dict[str, BasePredictor]:
        """Train multiple models"""
        trained_models = {}
        
        # Get models to train from factory
        models_to_train = ModelFactory.create_models(model_choice, model_size)
        
        for model_name, model in models_to_train.items():
            try:
                with st.spinner(f"Training {model_name}..."):
                    trained_models[model_name] = ModelTrainer.train_model(model, train_data)
                    st.success(f"{model_name} trained successfully!")
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                st.error(f"Error training {model_name}: {str(e)}")
                continue
                
        return trained_models