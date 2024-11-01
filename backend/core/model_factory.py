from typing import Dict, Any, Optional
from darts import TimeSeries
import logging

# Use absolute imports
from backend.models.base_model import BasePredictor
from backend.models.chronos_model import ChronosPredictor
from backend.models.nbeats_model import NBEATSPredictor
from backend.models.prophet_model import ProphetModel
from backend.models.tide_model import TiDEPredictor
from backend.models.time_mixer import TSMixerPredictor

logger = logging.getLogger(__name__)

class ModelFactory:
    @staticmethod
    def get_available_models() -> Dict[str, type]:
        return {
            "N-BEATS": NBEATSPredictor,
            "Prophet": ProphetModel,
            "TiDE": TiDEPredictor,
            "Chronos": ChronosPredictor,
            "TSMixer": TSMixerPredictor
        }

    @staticmethod
    def create_model(model_name: str, model_size: str = "small") -> BasePredictor:
        """
        Create a model instance based on the model name.
        Only Chronos model uses the model_size parameter.
        """
        models = ModelFactory.get_available_models()
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}")
            
        # Only pass model_size to Chronos
        if model_name == "Chronos":
            return models[model_name](model_size=model_size)
        else:
            return models[model_name]()

    @staticmethod
    def create_models(model_choice: str, model_size: str = "small") -> Dict[str, BasePredictor]:
        """Create multiple model instances based on the model choice."""
        if model_choice == "All Models":
            return {name: ModelFactory.create_model(name, model_size) 
                    for name in ModelFactory.get_available_models().keys()}
        else:
            return {model_choice: ModelFactory.create_model(model_choice, model_size)}

def train_models(
    train_data: TimeSeries,
    model_choice: str = "All Models",
    model_size: str = "small"
) -> Dict[str, BasePredictor]:
    """Unified training function for all models"""
    trained_models = {}
    
    # Determine which models to train
    if model_choice == "All Models":
        models_to_train = ModelFactory.get_available_models().keys()
    else:
        models_to_train = [model_choice]
    
    # Train selected models
    for model_name in models_to_train:
        try:
            with st.spinner(f"Training {model_name} model..."):
                model = ModelFactory.create_model(
                    model_name,
                    model_size=model_size if model_name == "Chronos" else None
                )
                model.train(train_data)
                trained_models[model_name] = model
                st.success(f"{model_name} model trained successfully!")
                
        except Exception as e:
            st.error(f"Error training {model_name} model: {str(e)}")
            logger.error(f"Error training {model_name} model: {str(e)}")
            continue
    
    return trained_models 