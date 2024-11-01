import pytest
from backend.models.data.forecasting import generate_forecasts
from darts import TimeSeries
from backend.models.data.training import train_models


def test_generate_forecasts(trained_models, sample_time_series):
    forecasts = generate_forecasts(
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
    train = sample_time_series[:80]
    return train_models(train, model_choice="All Models", model_size="small")
