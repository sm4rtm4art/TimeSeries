import logging

from backend.core.interfaces.base_model import TimeSeriesPredictor
from backend.domain.models.deep_learning.nbeats import NBEATSPredictor
from backend.domain.models.deep_learning.tide_model import TiDEPredictor
from backend.domain.models.deep_learning.time_mixer import TSMixerPredictor
from backend.domain.models.experimental.chronos_model import ChronosPredictor
from backend.domain.models.statistical.prophet import ProphetModel

logger = logging.getLogger(__name__)


class ModelFactory:
    @staticmethod
    def create_models(
        model_choice: str,
        model_size: str = "small",
    ) -> tuple[dict[str, TimeSeriesPredictor], dict[str, str]]:
        models = {}
        failed_models = {}
        logger.info(f"Creating models for choice: {model_choice}")

        model_map = {
            "N-BEATS": lambda: NBEATSPredictor(),
            "Prophet": lambda: ProphetModel(),
            "TiDE": lambda: TiDEPredictor(),
            "TSMixer": lambda: TSMixerPredictor(),
            "Chronos": lambda: ChronosPredictor(size=model_size),
        }

        if model_choice == "All Models":
            models_to_create = list(model_map.keys())[:-1]  # Exclude Chronos from All Models
        else:
            models_to_create = [model_choice]

        for model_name in models_to_create:
            try:
                if model_name in model_map:
                    models[model_name] = model_map[model_name]()
                    logger.info(f"Successfully initialized {model_name}")
            except Exception as e:
                error_msg = f"Failed to initialize {model_name}: {str(e)}"
                logger.error(error_msg)
                failed_models[model_name] = error_msg

        if not models:
            raise ValueError("No models could be initialized successfully")

        return models, failed_models
