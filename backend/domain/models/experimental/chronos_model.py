"""Chronos Model Implementation for Time Series Forecasting"""

import logging

import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Pipeline

from backend.core.interfaces.base_model import TimeSeriesPredictor

logger = logging.getLogger(__name__)


class ChronosPipeline(Pipeline):
    def __init__(self, model, tokenizer, device_map="auto"):
        super().__init__(model=model, tokenizer=tokenizer, device_map=device_map)

    def predict(self, sequences, prediction_length: int):
        inputs = self.tokenizer(sequences, return_tensors="pt", padding=True)
        outputs = self.model.generate(
            **inputs,
            max_length=prediction_length,
            num_return_sequences=1,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


class ChronosPredictor(TimeSeriesPredictor):
    def __init__(self, model_name: str = "Chronos", size: str = "small"):
        super().__init__(model_name)
        self.size = size
        self.model_sizes = {
            "tiny": "chronos-t5-tiny",
            "small": "chronos-t5-small",
            "medium": "chronos-t5-medium",
            "large": "chronos-t5-large",
        }
        self._initialize_pipeline()

    def _get_device_map(self) -> str:
        """Get the appropriate device map based on hardware."""
        config = self._get_hardware_config()
        return "cuda" if config["accelerator"] == "gpu" else "cpu"

    def _initialize_pipeline(self):
        try:
            logger.info(f"Initializing Chronos model with size: {self.size}")
            model_name = f"amazon/{self.model_sizes[self.size]}"

            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                device_map=self._get_device_map(),
                torch_dtype=torch.float32 if self._get_device_map() == "cpu" else torch.bfloat16,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            self.model = ChronosPipeline(model=model, tokenizer=tokenizer)
            logger.info(f"Successfully loaded pretrained model: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing Chronos pipeline: {str(e)}")
            raise

    def _train_model(self, scaled_data: TimeSeries, **kwargs) -> None:
        """Store training data for later use."""
        self.training_data = scaled_data
        logger.info("Training data stored successfully")

    def _generate_forecast(self, horizon: int) -> TimeSeries:
        """Generate forecast using the Chronos model."""
        try:
            input_sequence = self.training_data.values().tolist()
            predictions = self.model.predict(
                input_sequence,
                prediction_length=horizon,
            )
            return TimeSeries.from_values(np.array(predictions))
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            raise

    def _generate_historical_forecasts(
        self,
        series: TimeSeries,
        start: float,
        forecast_horizon: int,
        stride: int,
        retrain: bool,
    ) -> TimeSeries:
        """Generate historical forecasts for backtesting."""
        try:
            if isinstance(start, float):
                start_idx = int(len(series) * start)
            else:
                start_idx = series.get_index_at_point(start)

            predictions = []
            prediction_times = []

            for i in range(start_idx, len(series) - forecast_horizon + 1, stride):
                historical_data = series[:i].values().tolist()
                forecast = self.model.predict(
                    historical_data,
                    prediction_length=forecast_horizon,
                )
                predictions.append(forecast[0])
                prediction_times.append(series.time_index[i])

            return TimeSeries.from_times_and_values(
                times=pd.DatetimeIndex(prediction_times),
                values=predictions,
            )

        except Exception as e:
            logger.error(f"Error in historical forecasts: {str(e)}")
            raise
