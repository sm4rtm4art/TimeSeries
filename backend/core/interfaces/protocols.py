from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Optional, List
from darts import TimeSeries
import pandas as pd

class TimeSeriesValidator(ABC):
    """Protocol defining validation methods for time series models."""
    @abstractmethod
    def train(self, data: TimeSeries) -> None:
        """Train the model on the given time series data."""
        pass

    @abstractmethod
    def predict(self, horizon: int, data: Optional[TimeSeries] = None) -> TimeSeries:
        """Generate predictions for the specified horizon."""
        pass

    @abstractmethod
    def evaluate(self, actual: TimeSeries, predicted: TimeSeries) -> Dict[str, float]:
        """Evaluate model performance using various metrics."""
        pass 