from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union, Optional, Any
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, rmse, mse, mae
import logging
import traceback

logger = logging.getLogger(__name__)

class BasePredictor(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.scaler = Scaler()
        self.train_data = None

    @abstractmethod
    def _create_model(self) -> Any:
        """Create and return the specific model instance"""
        pass

    def train(self, data: TimeSeries) -> None:
        """Train the model on the provided data."""
        try:
            logger.info(f"Training {self.model_name} model")
            
            # Create model if not exists
            if self.model is None:
                self.model = self._create_model()
            
            # Convert data to float32 before any operations
            data_float32 = data.astype(np.float32)
            
            # Fit the scaler on the float32 data
            self.scaler.fit(data_float32)
            
            # Store training data
            self.train_data = data_float32
            
            # Scale and train the model
            scaled_data = self.scaler.transform(data_float32)
            self._train_model(scaled_data)
            
            self.is_trained = True
            logger.info(f"{self.model_name} training completed")
            
        except Exception as e:
            logger.error(f"Error in {self.model_name} training: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    @abstractmethod
    def _train_model(self, scaled_data: TimeSeries) -> None:
        """Implement specific model training logic"""
        pass

    def predict(self, horizon: int, data: Optional[TimeSeries] = None) -> TimeSeries:
        """Standard prediction implementation with scaling"""
        try:
            if not self.is_trained:
                raise ValueError(f"{self.model_name} must be trained before prediction")
            
            # Use provided data or training data
            input_data = data if data is not None else self.train_data
            if input_data is None:
                raise ValueError("No data provided for prediction")
            
            scaled_data = self.scaler.transform(input_data.astype(np.float32))
            
            # Generate and inverse scale forecast
            forecast = self._generate_forecast(horizon, scaled_data)
            return self.scaler.inverse_transform(forecast)
            
        except Exception as e:
            logger.error(f"Error in {self.model_name} prediction: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    @abstractmethod
    def _generate_forecast(self, horizon: int, scaled_data: TimeSeries) -> TimeSeries:
        """Implement specific model forecasting logic"""
        pass

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
                raise ValueError(f"{self.model_name} must be trained before generating historical forecasts")
            
            # Convert data to float32 and scale
            data_float32 = series.astype(np.float32)
            scaled_data = self.scaler.transform(data_float32)
            
            # Generate historical forecasts
            return self._generate_historical_forecasts(
                scaled_data=scaled_data,
                start=start,
                forecast_horizon=forecast_horizon,
                stride=stride,
                retrain=retrain,
                verbose=verbose,
                **kwargs
            )
            
        except Exception as e:
            logger.error(f"Error in {self.model_name} historical forecasts: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    @abstractmethod
    def _generate_historical_forecasts(
        self,
        scaled_data: TimeSeries,
        start: Union[pd.Timestamp, float],
        forecast_horizon: int,
        stride: int,
        retrain: bool = False,
        verbose: bool = False,
        **kwargs
    ) -> TimeSeries:
        """Implement specific historical forecasts logic"""
        pass

    def backtest(
        self,
        data: TimeSeries,
        start: Union[pd.Timestamp, float],
        forecast_horizon: int,
        stride: int = 1
    ) -> Dict[str, Union[TimeSeries, Dict[str, float]]]:
        """Standard backtesting implementation"""
        try:
            if not self.is_trained:
                raise ValueError(f"{self.model_name} must be trained before backtesting")
            
            # Generate historical forecasts
            historical_forecasts = self.historical_forecasts(
                series=data,
                start=start,
                forecast_horizon=forecast_horizon,
                stride=stride
            )
            
            # Calculate metrics
            actual_data = data[historical_forecasts.start_time():historical_forecasts.end_time()]
            metrics = {
                'MAPE': float(mape(actual_data, historical_forecasts)),
                'RMSE': float(rmse(actual_data, historical_forecasts)),
                'MSE': float(mse(actual_data, historical_forecasts)),
                'MAE': float(mae(actual_data, historical_forecasts))
            }
            
            logger.info(f"Generated metrics for {self.model_name}: {metrics}")
            
            return {
                'backtest': historical_forecasts,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Error in {self.model_name} backtesting: {str(e)}")
            logger.error(traceback.format_exc())
            raise