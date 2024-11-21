"""
Chronos model
"""
import numpy as np
import pandas as pd
import streamlit as st
import torch
from chronos import ChronosPipeline
from darts import TimeSeries
from typing import Dict, Tuple, Union, Optional
import logging
import traceback
from darts.metrics import mape, mse, rmse, mae
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import MinMaxScaler
from ....core.interfaces.base_model import TimeSeriesPredictor

logger = logging.getLogger(__name__)

class ChronosPredictor(TimeSeriesPredictor):
    def __init__(
        self,
        size: str = "small",
        **kwargs
    ):
        """Initialize the Chronos predictor with pretrained models."""
        super().__init__()
        self.size = size  # tiny, mini, small, base, or large
        self.model_name = "Chronos"
        self.model = None
        self.scaler = Scaler(scaler=MinMaxScaler(feature_range=(-1, 1)))
        self.input_chunk_length = kwargs.get('input_chunk_length', 24)
        
        # Map size to model names according to Chronos paper
        self.model_sizes = {
            "tiny": "chronos-t5-tiny",    # 8M parameters
            "mini": "chronos-t5-mini",    # 20M parameters
            "small": "chronos-t5-small",  # 46M parameters
            "base": "chronos-t5-base",    # 200M parameters
            "large": "chronos-t5-large"   # 710M parameters
        }
        
        if size not in self.model_sizes:
            raise ValueError(f"Invalid size. Must be one of {list(self.model_sizes.keys())}")
            
        self._initialize_pipeline()

    def _get_device_map(self):
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _initialize_pipeline(self):
        """Initialize the Chronos pipeline with the appropriate model."""
        try:
            logger.info(f"Initializing Chronos model with size: {self.size}")
            model_name = f"amazon/{self.model_sizes[self.size]}"
            self.model = ChronosPipeline.from_pretrained(
                model_name,
                device_map=self._get_device_map(),
                torch_dtype=torch.float32 if self._get_device_map() == "cpu" else torch.bfloat16,
            )
            logger.info(f"Successfully loaded pretrained model: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing Chronos pipeline: {str(e)}")
            raise

    def train(self, data: TimeSeries) -> None:
        """Store training data and fit scaler."""
        logger.info("Training Chronos model")
        try:
            # Scale the data between -1 and 1
            self.training_data = self.scaler.fit_transform(data)
            logger.info("Data scaled successfully")
            self.is_trained = True
        except Exception as e:
            logger.error(f"Error in training: {str(e)}")
            raise

    def predict(self, horizon: int, data: TimeSeries = None) -> TimeSeries:
        """Generate predictions."""
        try:
            if data is None:
                data = self.training_data
            else:
                # Scale new data using fitted scaler
                data = self.scaler.transform(data)
            
            # Convert to tensor and predict
            context = torch.tensor(data.values())
            forecast = self.model.predict(
                context=context,
                prediction_length=horizon,
                num_samples=20
            )
            
            # Take median of samples and convert back to original scale
            predictions = np.median(forecast[0].numpy(), axis=0)
            forecast_dates = pd.date_range(
                start=data.time_index[-1] + data.freq,
                periods=horizon,
                freq=data.freq
            )
            
            forecast_ts = TimeSeries.from_times_and_values(
                times=forecast_dates,
                values=predictions.reshape(-1, 1)
            )
            
            # Inverse transform to original scale
            return self.scaler.inverse_transform(forecast_ts)
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise

    def backtest(
        self,
        data: TimeSeries,
        start: Union[pd.Timestamp, float],
        forecast_horizon: int,
        stride: int = 1,
        **kwargs
    ) -> Dict[str, Union[TimeSeries, Dict[str, float]]]:
        """Perform backtesting on the model."""
        try:
            logger.info(f"Starting Chronos backtest. Data length: {len(data)}, Forecast horizon: {forecast_horizon}, Start: {start}")
            
            # Convert float start to index if needed
            if isinstance(start, float):
                start = int(len(data) * start)
            start_timestamp = data.time_index[start]
            
            # Generate historical forecasts
            historical_forecasts = []
            for i in range(start, len(data) - forecast_horizon + 1, stride):
                # Get historical context
                context = data[:i]
                
                # Generate forecast
                forecast = self.predict(forecast_horizon, context)
                historical_forecasts.append(forecast)
            
            # Combine forecasts
            combined_forecasts = pd.concat([f.pd_series() for f in historical_forecasts])
            historical_forecasts = TimeSeries.from_series(combined_forecasts)
            
            # Get actual data for the forecast period
            actual_data = data.slice(start_timestamp, data.end_time())
            
            # Calculate metrics
            from backend.utils.metrics import calculate_metrics
            metrics = calculate_metrics(actual_data, historical_forecasts)
            
            logger.info(f"Backtest completed. Forecast length: {len(historical_forecasts)}")
            logger.info(f"Backtest metrics: {metrics}")
            
            return {
                'backtest': historical_forecasts,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Error in Chronos backtesting: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def historical_forecasts(
        self,
        series: TimeSeries,
        start: Union[pd.Timestamp, float],
        forecast_horizon: int,
        stride: int = 1,
        retrain: bool = False,
        verbose: bool = False,
        **kwargs
    ) -> TimeSeries:
        """Generate historical forecasts."""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before generating historical forecasts")
                
            logger.info(f"Starting historical forecasts. Series length: {len(series)}")
            
            # Convert float start to index if needed
            if isinstance(start, float):
                start_idx = int(len(series) * start)
            else:
                start_idx = series.time_index.get_loc(start)
            
            # Scale the values manually instead of using Darts scaler
            values = series.values().flatten()
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaled_values = scaler.fit_transform(values.reshape(-1, 1))
            
            predictions = []
            prediction_times = []
            
            for i in range(start_idx, len(series) - forecast_horizon + 1, stride):
                # Get historical context
                context = scaled_values[:i]
                context_tensor = torch.tensor(context)
                
                # Generate forecast
                forecast = self.model.predict(
                    context=context_tensor,
                    prediction_length=forecast_horizon,
                    num_samples=20
                )
                
                # Store median prediction
                pred = np.median(forecast[0].numpy(), axis=0)
                predictions.append(pred)
                
                # Store forecast dates
                pred_times = pd.date_range(
                    start=series.time_index[i],
                    periods=forecast_horizon,
                    freq=series.freq
                )
                prediction_times.extend(pred_times)
            
            # Combine all predictions
            all_predictions = np.concatenate(predictions)
            
            # Inverse transform predictions
            all_predictions_reshaped = scaler.inverse_transform(
                all_predictions.reshape(-1, 1)
            ).flatten()
            
            # Create final TimeSeries
            return TimeSeries.from_times_and_values(
                times=pd.DatetimeIndex(prediction_times),
                values=all_predictions_reshaped.reshape(-1, 1)
            )
            
        except Exception as e:
            logger.error(f"Error in historical_forecasts: {str(e)}")
            logger.error(traceback.format_exc())
            raise

def train_chronos_model(train_data: TimeSeries, model_size: str = "small") -> ChronosPredictor:
    """
    Train a Chronos model using the ChronosPredictor class.
    
    Args:
    train_data (TimeSeries): The training data as a Darts TimeSeries object.
    model_size (str): Size of the Chronos model. Options: "tiny", "small", "medium", "large".
    
    Returns:
    ChronosPredictor: A trained ChronosPredictor instance.
    """
    st.text(f"Training Chronos model (size: {model_size})...")
    model = ChronosPredictor(model_size=model_size)
    try:
        model.train(train_data)
        st.text("Chronos model training completed")
    except Exception as e:
        st.error(f"Error during Chronos model training: {str(e)}")
        raise
    return model


def make_chronos_forecast(model: ChronosPredictor, data: TimeSeries, forecast_horizon: int) -> TimeSeries:
    return model.predict(forecast_horizon, data)
