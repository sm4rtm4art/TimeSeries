"""Custom exceptions for the application."""


class ModelTrainingError(Exception):
    """Raised when there is an error during model training."""

    pass


class DataProcessingError(Exception):
    """Raised when there is an error during data processing."""

    pass


class ForecastingError(Exception):
    """Raised when there is an error during forecasting."""

    pass
