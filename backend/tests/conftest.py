import pytest
from pytest_mock import MockerFixture
from darts import TimeSeries
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
