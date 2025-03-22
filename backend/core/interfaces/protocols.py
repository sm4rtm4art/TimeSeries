from abc import ABC, abstractmethod

from darts import TimeSeries


class TimeSeriesValidator(ABC):
    """Protocol defining validation methods for time series models."""

    @abstractmethod
    def train(self, data: TimeSeries) -> None:
        """Train the model on the given time series data."""
        pass

    @abstractmethod
    def predict(self, horizon: int, data: TimeSeries | None = None) -> TimeSeries:
        """Generate predictions for the specified horizon."""
        pass

    @abstractmethod
    def evaluate(self, actual: TimeSeries, predicted: TimeSeries) -> dict[str, float]:
        """Evaluate model performance using various metrics."""
        pass
