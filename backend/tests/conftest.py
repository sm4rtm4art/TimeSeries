import pytest
from pytest_mock import MockerFixture
from darts import TimeSeries
import numpy as np
import pandas as pd
from backend.app.streamlit import DataHandler, TimeSeriesForecastApp

@pytest.fixture
def mock_data_loader(mocker: MockerFixture):
    return mocker.Mock()

@pytest.fixture
def data_handler(mock_data_loader):
    return DataHandler(mock_data_loader)

@pytest.fixture
def app(data_handler):
    return TimeSeriesForecastApp(data_handler)

@pytest.fixture
def mock_time_series(mocker: MockerFixture):
    return mocker.Mock(spec=TimeSeries)

@pytest.fixture
def sample_time_series():
    """Create a sample time series for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    values = np.random.randn(100, 1)
    return TimeSeries.from_times_and_values(dates, values)

@pytest.fixture
def train_test_data(sample_time_series):
    """Split sample time series into train and test sets."""
    train = sample_time_series[:80]
    test = sample_time_series[80:]
    return train, test
