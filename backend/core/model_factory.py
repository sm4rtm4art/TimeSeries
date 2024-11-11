from typing import Dict, Type
from backend.core.interfaces.model import TimeSeriesPredictor
from backend.domain.models.statistical.prophet import ProphetModel
from backend.domain.models.deep_learning.tide_model import TiDEPredictor
from backend.domain.models.deep_learning.nbeats import NBEATSPredictor
from backend.domain.models.deep_learning.time_mixer import TSMixerModel

class ModelFactory:
    _models: Dict[str, Type[TimeSeriesPredictor]] = {
        "Prophet": ProphetModel,
        "N-BEATS": NBEATSPredictor,
        "TiDE": TiDEPredictor,
        "TSMixer": TSMixerModel
    }

    @classmethod
    def create(cls, model_name: str, **kwargs) -> TimeSeriesPredictor:
        if model_name not in cls._models:
            raise ValueError(f"Unknown model: {model_name}")
        return cls._models[model_name](**kwargs)