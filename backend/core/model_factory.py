import logging

from backend.core.interfaces.base_model import TimeSeriesPredictor
from backend.domain.models.deep_learning.nbeats import NBEATSPredictor
from backend.domain.models.deep_learning.nhits_model import NHiTSPredictor
from backend.domain.models.deep_learning.tide_model import TiDEPredictor
from backend.domain.models.deep_learning.time_mixer import TSMixerPredictor
from backend.domain.models.experimental.chronos_model import ChronosPredictor
from backend.domain.models.statistical.arima_model import ARIMAPredictor
from backend.domain.models.statistical.ets_model import ETSPredictor
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
            "N-HiTS": lambda: NHiTSPredictor(),
            "Prophet": lambda: ProphetModel(),
            "TiDE": lambda: TiDEPredictor(),
            "TSMixer": lambda: TSMixerPredictor(),
            "ARIMA": lambda: ARIMAPredictor(),
            "ETS": lambda: ETSPredictor(),
            "Chronos": lambda: ChronosPredictor(size=model_size),
        }

        all_model_keys = ["N-BEATS", "N-HiTS", "Prophet", "TiDE", "TSMixer", "ARIMA", "ETS"]

        if model_choice == "All Models":
            models_to_create = all_model_keys
        elif model_choice in model_map:
            models_to_create = [model_choice]
        else:
            logger.warning(f"Unknown model choice: {model_choice}")
            models_to_create = []

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
