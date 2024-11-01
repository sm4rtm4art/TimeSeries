"""
NGBoost Model for Probabilistic Time Series Forecasting
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from darts import TimeSeries
from ngboost import NGBRegressor
from ngboost.distns import Normal
import logging
from backend.models.feature_extraction import TimeSeriesFeatureExtractor

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class NGBoostPredictor:
    def __init__(
        self,
        lookback: int = 30,
        n_estimators: int = 500,
        learning_rate: float = 0.01
    ):
        """
        Initialize NGBoost predictor.
        
        Args:
            lookback: Number of past time steps to use
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate
        """
        self.lookback = lookback
        self.model = NGBRegressor(
            Dist=Normal,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            natural_gradient=True
        )
        self.feature_extractor = TimeSeriesFeatureExtractor()
        self.is_trained = False
        self.model_name = "NGBoost"
        
    def train(self, data: TimeSeries):
        try:
            X_train, y_train = self._create_supervised_data(data)
            
            self.model.fit(X_train, y_train)
            self.is_trained = True
            logger.info("NGBoost training completed successfully")
            
        except Exception as e:
            logger.error(f"Error in NGBoost training: {str(e)}")
            raise

    def predict(
        self, 
        horizon: int, 
        data: Optional[TimeSeries] = None,
        num_samples: int = 100
    ) -> Tuple[TimeSeries, TimeSeries, TimeSeries]:
        """
        Generate probabilistic predictions.
        
        Returns:
            Tuple of (mean forecast, lower CI, upper CI)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        try:
            last_window = data[-self.lookback:] if data is not None else self.training_data[-self.lookback:]
            future_dates = pd.date_range(
                start=last_window.time_index[-1] + pd.Timedelta(days=1),
                periods=horizon,
                freq=last_window.freq_str
            )
            
            predictions_dist = []
            current_window = last_window.values().flatten()
            
            for future_date in future_dates:
                features = self.feature_extractor.create_features(pd.DatetimeIndex([future_date]))
                X = np.concatenate([current_window[-self.lookback:], features.iloc[0].values])
                
                dist_params = self.model.pred_dist(X.reshape(1, -1))
                mean = dist_params.mean()[0]
                std = dist_params.std()[0]
                
                samples = np.random.normal(mean, std, num_samples)
                predictions_dist.append((mean, mean - 1.96*std, mean + 1.96*std))
                current_window = np.append(current_window[1:], mean)
            
            means, lower_ci, upper_ci = zip(*predictions_dist)
            
            return (
                TimeSeries.from_times_and_values(future_dates, means),
                TimeSeries.from_times_and_values(future_dates, lower_ci),
                TimeSeries.from_times_and_values(future_dates, upper_ci)
            )
            
        except Exception as e:
            logger.error(f"Error in NGBoost prediction: {str(e)}")
            raise

    def get_calibration_score(self) -> float:
        """Calculate prediction interval calibration score."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Implementation of calibration score calculation
        return self.model.score(self.X_val, self.y_val)

