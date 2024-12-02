"""
XGBoost Model for Time Series Forecasting
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from darts import TimeSeries
from xgboost import XGBRegressor
import logging
from backend.models.feature_extraction import TimeSeriesFeatureExtractor

logger = logging.getLogger(__name__)

class XGBoostPredictor:
    def __init__(
        self,
        lookback: int = 30,
        n_estimators: int = 1000,
        learning_rate: float = 0.01,
        max_depth: int = 5
    ):
        """
        Initialize XGBoost predictor.
        
        Args:
            lookback: Number of past time steps to use
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate
            max_depth: Maximum tree depth
        """
        self.lookback = lookback
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            objective='reg:squarederror',
            tree_method='hist'
        )
        self.feature_extractor = TimeSeriesFeatureExtractor()
        self.scaler = None
        self.is_trained = False
        self.model_name = "XGBoost"
        
    def _create_supervised_data(
        self, 
        data: TimeSeries, 
        is_train: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        values = data.values().flatten()
        dates = data.time_index
        features = self.feature_extractor.create_features(dates)
        
        X, y = [], []
        for i in range(self.lookback, len(values)):
            X.append(np.concatenate([
                values[i-self.lookback:i],
                features.iloc[i].values
            ]))
            if is_train:
                y.append(values[i])
                
        X = np.array(X)
        y = np.array(y) if is_train else None
        return X, y
        
    def train(self, data: TimeSeries):
        try:
            logger.info("Preparing data for XGBoost training")
            X_train, y_train = self._create_supervised_data(data)
            
            logger.info("Training XGBoost model")
            self.model.fit(
                X_train, 
                y_train,
                eval_set=[(X_train, y_train)],
                early_stopping_rounds=50,
                verbose=False
            )
            
            self.is_trained = True
            logger.info("XGBoost training completed successfully")
            
        except Exception as e:
            logger.error(f"Error in XGBoost training: {str(e)}")
            raise

    def predict(self, horizon: int, data: Optional[TimeSeries] = None) -> TimeSeries:
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        try:
            last_window = data[-self.lookback:] if data is not None else self.training_data[-self.lookback:]
            future_dates = pd.date_range(
                start=last_window.time_index[-1] + pd.Timedelta(days=1),
                periods=horizon,
                freq=last_window.freq_str
            )
            
            predictions = []
            current_window = last_window.values().flatten()
            
            for future_date in future_dates:
                features = self.feature_extractor.create_features(pd.DatetimeIndex([future_date]))
                X = np.concatenate([current_window[-self.lookback:], features.iloc[0].values])
                pred = self.model.predict(X.reshape(1, -1))[0]
                predictions.append(pred)
                current_window = np.append(current_window[1:], pred)
            
            return TimeSeries.from_times_and_values(future_dates, predictions)
            
        except Exception as e:
            logger.error(f"Error in XGBoost prediction: {str(e)}")
            raise

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
            
        feature_names = (
            [f'lag_{i}' for i in range(1, self.lookback + 1)] +
            self.feature_extractor.get_feature_names()
        )
        
        importance_scores = self.model.feature_importances_
        return dict(zip(feature_names, importance_scores))
