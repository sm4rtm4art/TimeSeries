"""
Feature Extraction for Time Series Models
"""

import numpy as np
import pandas as pd
from typing import List, Optional

class TimeSeriesFeatureExtractor:
    def __init__(self, seasonal_periods: Optional[List[int]] = None):
        self.seasonal_periods = seasonal_periods or [7, 30, 365]
        
    def create_features(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Create time-based features."""
        features = pd.DataFrame(index=dates)
        
        # Time-based features
        features['hour'] = dates.hour
        features['day'] = dates.day
        features['month'] = dates.month
        features['year'] = dates.year
        features['dayofweek'] = dates.dayofweek
        features['quarter'] = dates.quarter
        
        # Cyclical features
        features['hour_sin'] = np.sin(2 * np.pi * dates.hour/24)
        features['hour_cos'] = np.cos(2 * np.pi * dates.hour/24)
        features['month_sin'] = np.sin(2 * np.pi * dates.month/12)
        features['month_cos'] = np.cos(2 * np.pi * dates.month/12)
        features['dayofweek_sin'] = np.sin(2 * np.pi * dates.dayofweek/7)
        features['dayofweek_cos'] = np.cos(2 * np.pi * dates.dayofweek/7)
        
        return features
        
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return [
            'hour', 'day', 'month', 'year', 'dayofweek', 'quarter',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
            'dayofweek_sin', 'dayofweek_cos'
        ]

