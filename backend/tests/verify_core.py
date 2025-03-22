import numpy as np
import pandas as pd
import pytest
from darts import TimeSeries

from backend.core.factory import ModelFactory
from backend.domain.services.forecasting import ForecastingService
from backend.domain.services.preprocessing import TimeSeriesPreprocessor


@pytest.fixture
def sample_data():
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    values = np.random.randn(100, 1)
    return TimeSeries.from_times_and_values(dates, values)


def test_core_pipeline(sample_data):
    """Test the entire pipeline with the new architecture"""
    try:
        # 1. Preprocessing
        preprocessor = TimeSeriesPreprocessor()
        processed_data, metadata = preprocessor.preprocess(sample_data)
        assert isinstance(processed_data, TimeSeries)
        assert isinstance(metadata, dict)

        # 2. Model Creation
        model = ModelFactory.create("Prophet")
        assert model is not None

        # 3. Training
        model.train(processed_data)
        assert model.is_trained

        # 4. Forecasting
        forecast_service = ForecastingService()
        forecasts = forecast_service.generate_forecasts(
            trained_models={"Prophet": model},
            data=processed_data,
            forecast_horizon=10,
            backtests={},
        )
        assert "Prophet" in forecasts
        assert isinstance(forecasts["Prophet"]["future"], TimeSeries)

    except Exception as e:
        pytest.fail(f"Pipeline test failed: {str(e)}")
