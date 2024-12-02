import pytest
from darts import TimeSeries
from backend.domain.services.forecasting import ForecastingService
from backend.domain.services.training import ModelTrainingService


def test_generate_forecasts(trained_models, sample_time_series):
    forecasts = ForecastingService().generate_forecasts(
        trained_models,
        sample_time_series,
        forecast_horizon=20
    )
    assert len(forecasts) > 0
    for forecast in forecasts.values():
        assert isinstance(forecast, TimeSeries)
        assert len(forecast) == 20


@pytest.fixture
def trained_models(sample_time_series):
    """Create trained models for testing."""
    training_service = ModelTrainingService()
    return training_service.train_models(sample_time_series, model_choice="All Models", model_size="small")
