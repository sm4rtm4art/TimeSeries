import numpy as np
import pandas as pd
import pytest
from darts import TimeSeries

from backend.domain.models.deep_learning.nbeats import NBEATSPredictor
from backend.domain.models.deep_learning.tide_model import TiDEPredictor
from backend.domain.models.deep_learning.time_mixer import TSMixerModel as TSMixerPredictor
from backend.domain.models.statistical.prophet import ProphetModel as ProphetPredictor


# Test data fixture
@pytest.fixture
def train_test_data():
    """Create sample train/test data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    values = np.random.randn(100, 1)
    ts = TimeSeries.from_times_and_values(dates, values)
    train = ts[:80]
    test = ts[80:]
    return train, test


# Individual model tests
class TestNBEATSPredictor:
    def test_initialization(self):
        model = NBEATSPredictor()
        assert model is not None
        assert model.model_name == "N-BEATS"
        assert not model.is_trained

    def test_training(self, train_test_data):
        train, _ = train_test_data
        model = NBEATSPredictor()
        model.train(train)
        assert model.is_trained

    def test_forecasting(self, train_test_data):
        train, test = train_test_data
        model = NBEATSPredictor()
        model.train(train)
        forecast = model.forecast(horizon=len(test))
        assert isinstance(forecast, TimeSeries)
        assert len(forecast) == len(test)

    def test_backtest(self, train_test_data):
        train, _ = train_test_data
        model = NBEATSPredictor()
        model.train(train)
        backtest = model.backtest(
            data=train,
            start=0.5,
            forecast_horizon=5,
            stride=1,
        )
        assert isinstance(backtest, TimeSeries)


class TestProphetPredictor:
    def test_initialization(self):
        model = ProphetPredictor()
        assert model is not None
        assert model.model_name == "Prophet"
        assert not model.is_trained

    def test_training(self, train_test_data):
        train, _ = train_test_data
        model = ProphetPredictor()
        model.train(train)
        assert model.is_trained

    def test_forecasting(self, train_test_data):
        train, test = train_test_data
        model = ProphetPredictor()
        model.train(train)
        forecast = model.forecast(horizon=len(test))
        assert isinstance(forecast, TimeSeries)
        assert len(forecast) == len(test)


class TestTiDEPredictor:
    def test_initialization(self):
        model = TiDEPredictor()
        assert model is not None
        assert model.model_name == "TiDE"
        assert not model.is_trained

    def test_training(self, train_test_data):
        train, _ = train_test_data
        model = TiDEPredictor()
        model.train(train)
        assert model.is_trained

    def test_forecasting(self, train_test_data):
        train, test = train_test_data
        model = TiDEPredictor()
        model.train(train)
        forecast = model.forecast(horizon=len(test))
        assert isinstance(forecast, TimeSeries)
        assert len(forecast) == len(test)


class TestTSMixerPredictor:
    def test_initialization(self):
        model = TSMixerPredictor()
        assert model is not None
        assert model.model_name == "TSMixer"
        assert not model.is_trained

    def test_training(self, train_test_data):
        train, _ = train_test_data
        model = TSMixerPredictor()
        model.train(train)
        assert model.is_trained

    def test_forecasting(self, train_test_data):
        train, test = train_test_data
        model = TSMixerPredictor()
        model.train(train)
        forecast = model.forecast(horizon=len(test))
        assert isinstance(forecast, TimeSeries)
        assert len(forecast) == len(test)


# Common error handling tests
def test_untrained_model_error():
    """Test that untrained models raise appropriate errors."""
    models = [
        NBEATSPredictor(),
        ProphetPredictor(),
        TiDEPredictor(),
        TSMixerPredictor(),
    ]

    for model in models:
        with pytest.raises(ValueError, match="must be trained before"):
            model.forecast(horizon=10)
