import pytest

from backend.app.streamlit import DataHandler, DataHandlingError
from backend.data.data_loader import DataLoader


@pytest.fixture
def mock_data_loader(mocker):
    return mocker.Mock(spec=DataLoader)


@pytest.fixture
def data_handler(mock_data_loader):
    return DataHandler(mock_data_loader)


def test_load_and_process_data_success(data_handler, mock_data_loader, mocker):
    mock_data = mocker.Mock()
    mock_data.__len__ = mocker.Mock(return_value=100)
    mock_data.__getitem__ = mocker.Mock(side_effect=lambda x: mock_data)
    mock_data_loader.load_data.return_value = mock_data

    result = data_handler.load_and_process_data()
    assert isinstance(result, tuple)
    assert len(result) == 3
    data, train_data, test_data = result
    assert data == mock_data
    assert train_data == mock_data
    assert test_data == mock_data


def test_load_and_process_data_failure(data_handler, mock_data_loader):
    mock_data_loader.load_data.side_effect = Exception("Data loading failed")
    with pytest.raises(DataHandlingError, match="Error loading or processing data: Data loading failed"):
        data_handler.load_and_process_data()
