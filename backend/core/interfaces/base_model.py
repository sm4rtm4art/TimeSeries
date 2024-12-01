from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Optional, List
from darts import TimeSeries
import pandas as pd
import logging
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, rmse, mse
from backend.utils.model_utils import get_training_config

logger = logging.getLogger(__name__)

class TimeSeriesPredictor(ABC):
    """Base interface for all time series prediction models."""
    
    def __init__(self):
        self.is_trained = False
        self.model_name = self.__class__.__name__
        self.scaler = Scaler()

    @abstractmethod
    def train(self, data: TimeSeries) -> None:
        """Train the model on the given time series data."""
        pass

    @abstractmethod
    def predict(self, n: int) -> TimeSeries:
        """
        Generate predictions for n steps ahead.
        
        Args:
            n (int): Number of steps to forecast
            
        Returns:
            TimeSeries: Forecasted values
        """
        pass

    @abstractmethod
    def backtest(
        self,
        data: TimeSeries,
        start: Union[pd.Timestamp, float],
        forecast_horizon: int,
        stride: int = 1
    ) -> Dict[str, Union[TimeSeries, Dict[str, float]]]:
        """Perform backtesting on historical data."""
        pass

    def cross_validate(
        self,
        data: TimeSeries,
        num_folds: int = 5,
        horizon: int = 24,
        overlap_size: int = 0
    ) -> Dict[str, Union[List[float], Dict[str, float]]]:
        """
        Perform time series cross-validation with optional overlapping folds.
        """
        total_length = len(data)
        fold_size = (total_length - horizon) // num_folds
        
        if fold_size <= horizon:
            raise ValueError("Data length insufficient for specified parameters")
        
        metrics_per_fold = []
        forecasts = []
        
        for fold in range(num_folds):
            try:
                # Calculate fold boundaries and perform validation
                start_idx = fold * (fold_size - overlap_size)
                end_idx = start_idx + fold_size + horizon
                
                if end_idx > total_length:
                    break
                    
                train_data = data[:start_idx + fold_size]
                test_data = data[start_idx + fold_size:end_idx]
                
                self.train(train_data)
                forecast = self.predict(len(test_data))
                forecasts.append(forecast)
                
                # Use the same metrics as in backtest
                actual = test_data
                metrics = {
                    'MAPE': float(mape(actual, forecast)),
                    'RMSE': float(rmse(actual, forecast)),
                    'MSE': float(mse(actual, forecast))
                }
                
                metrics_per_fold.append(metrics)
                logger.info(f"Fold {fold + 1} metrics: {metrics}")
                
            except Exception as e:
                logger.error(f"Error in fold {fold + 1}: {str(e)}")
                continue
        
        return self._calculate_aggregate_metrics(metrics_per_fold, forecasts)

class DartsModelPredictor(TimeSeriesPredictor):
    """Base class for all Darts-based time series models."""
    def __init__(self):
        super().__init__()
        self.input_chunk_length = 24
        self.output_chunk_length = 12
        self.model = self._create_model()
        
    @abstractmethod
    def _create_model(self) -> Any:
        """Create and return the specific model implementation."""
        pass

    def train(self, data: TimeSeries) -> None:
        try:
            logger.info(f"Training {self.model_name}")
            scaled_data = self.scaler.fit_transform(data)
            self._train_model(scaled_data)
            self.is_trained = True
            logger.info(f"{self.model_name} trained successfully")
        except Exception as e:
            logger.error(f"Error training {self.model_name}: {str(e)}")
            raise

    def _train_model(self, data: TimeSeries) -> None:
        training_config = get_training_config()
        # Convert data to appropriate dtype before training
        data = data.astype(training_config['force_dtype'])
        self.model.fit(data)

    def predict(self, n: int) -> TimeSeries:
        try:
            if not self.is_trained:
                raise ValueError(f"{self.model_name} must be trained before prediction")
            forecast = self.model.predict(n=n)
            return self.scaler.inverse_transform(forecast)
        except Exception as e:
            logger.error(f"Error in {self.model_name} prediction: {str(e)}")
            raise

    def backtest(
        self,
        data: TimeSeries,
        start: Union[pd.Timestamp, float],
        forecast_horizon: int,
        stride: int = 1
    ) -> Dict[str, Union[TimeSeries, Dict[str, float]]]:
        try:
            scaled_data = self.scaler.transform(data)
            forecasts = self.model.historical_forecasts(
                series=scaled_data,
                start=start,
                forecast_horizon=forecast_horizon,
                stride=stride,
                retrain=False,
                verbose=False
            )
            
            forecasts = self.scaler.inverse_transform(forecasts)
            actual = data[forecasts.start_time():forecasts.end_time()]
            
            return {
                'backtest': forecasts,
                'metrics': {
                    'MAPE': float(mape(actual, forecasts)),
                    'RMSE': float(rmse(actual, forecasts)),
                    'MSE': float(mse(actual, forecasts))
                }
            }
        except Exception as e:
            logger.error(f"Error in {self.model_name} backtesting: {str(e)}")
            raise