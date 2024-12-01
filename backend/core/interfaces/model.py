from abc import ABC, abstractmethod
from typing import Dict, Optional
from darts import TimeSeries

class TimeSeriesPredictor(ABC):
    """Base interface for all time series prediction models."""
    
    def __init__(self):
        self.is_trained = False
        self.model_name = self.__class__.__name__

    @abstractmethod
    def train(self, data: TimeSeries) -> None:
        """Train the model on the given time series data."""
        pass

    @abstractmethod
    def predict(self, horizon: int, data: Optional[TimeSeries] = None) -> TimeSeries:
        """Generate predictions for the specified horizon."""
        pass

    @abstractmethod
    def backtest(
        self,
        data: TimeSeries,
        start: float,
        forecast_horizon: int
    ) -> Dict:
        """Perform backtesting on historical data."""
        pass 