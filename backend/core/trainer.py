import logging
import traceback

import numpy as np
from darts import TimeSeries

from backend.core.interfaces.base_model import TimeSeriesPredictor
from backend.core.model_factory import ModelFactory

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
            logger.info(
                f"Converting data to float32. Original dtype: {train_data.dtype}, New dtype: {train_data_float32.dtype}",
            )

            model.train(train_data_float32)
            return model
        except Exception as e:
            logger.error(f"Error training {model.model_name}: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    @staticmethod
    def train_models(
        train_data: TimeSeries,
        model_choice: str,
        model_size: str = "small",
    ) -> dict[str, TimeSeriesPredictor]:
        """Train selected models with the provided data."""
        try:
            logger.info(f"Creating models with choice: {model_choice}, size: {model_size}")
            models_to_train = ModelFactory.create_models(model_choice, model_size)

            # Unpack tuple if needed (models, failed_models)
            if isinstance(models_to_train, tuple):
                models_to_train, failed_models = models_to_train
                if failed_models:
                    for model_name, error in failed_models.items():
                        logger.warning(f"Failed to initialize {model_name}: {error}")

            logger.info(f"Models created: {list(models_to_train.keys()) if models_to_train else 'None'}")

            # Train each model
            for name, model in models_to_train.items():
                logger.info(f"Training {name}...")
                model.train(train_data)
                logger.info(f"{name} trained successfully")

            return models_to_train

        except Exception as e:
            logger.error(f"Error in train_models: {str(e)}")
            logger.error(traceback.format_exc())
            raise
