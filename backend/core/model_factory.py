from typing import Dict
from backend.core.interfaces.base_model import TimeSeriesPredictor
from backend.domain.models.deep_learning.nbeats import NBEATSPredictor
from backend.domain.models.deep_learning.tide_model import TiDEPredictor
from backend.domain.models.deep_learning.time_mixer import TSMixerPredictor
from backend.domain.models.statistical.prophet import ProphetModel
from backend.domain.models.experimental.chronos_model import ChronosPredictor
import logging

logger = logging.getLogger(__name__)

class ModelFactory:
    @staticmethod
    def create_models(model_choice: str, model_size: str = "small") -> Dict[str, TimeSeriesPredictor]:
        models = {}
        logger.info(f"Creating models for choice: {model_choice}")
        
        try:
            if model_choice == "All Models" or model_choice == "N-BEATS":
                models["N-BEATS"] = NBEATSPredictor()
            if model_choice == "All Models" or model_choice == "Prophet":
                models["Prophet"] = ProphetModel()
            if model_choice == "All Models" or model_choice == "TiDE":
                models["TiDE"] = TiDEPredictor()
            if model_choice == "All Models" or model_choice == "TSMixer":
                models["TSMixer"] = TSMixerPredictor()
            if model_choice == "All Models" or model_choice == "Chronos":
                models["Chronos"] = ChronosPredictor(size=model_size)
                
            return models
            
        except Exception as e:
            logger.error(f"Error creating models: {str(e)}")
            raise