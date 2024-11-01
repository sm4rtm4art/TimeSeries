import pytest
from backend.models.data.training import train_models
from backend.models.data.forecasting import generate_forecasts
from backend.utils.metrics import calculate_metrics
from darts import TimeSeries
from backend.models.data.evaluation import evaluate_models

@pytest.fixture
def trained_models(sample_time_series):
    train = sample_time_series[:80]
    models = train_models(train, model_choice="All Models", model_size="small")
    return models

def test_train_models(sample_time_series):
    train = sample_time_series[:80]
    models = train_models(train, model_choice="All Models", model_size="small")
    assert len(models) > 0
    for model in models.values():
        assert model.is_trained

def test_evaluate_models(sample_time_series):
    """Test model evaluation functionality."""
    train = sample_time_series[:80]
    test = sample_time_series[80:]
    
    # Train models
    trained_models = train_models(train, model_choice="All Models", model_size="small")
    
    # Evaluate models
    metrics = evaluate_models(trained_models, test)
    
    assert isinstance(metrics, dict)
    assert all(model in metrics for model in trained_models.keys())
    assert all(isinstance(metric_dict, dict) for metric_dict in metrics.values())

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
