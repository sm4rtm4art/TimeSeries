import numpy as np
import pandas as pd
import pytest

from backend.app.streamlit import TimeSeries


@pytest.mark.parametrize(
    "model_choice, forecast_horizon",
    [
        ("All Models", 30),
        ("N-BEATS", 60),
        ("Prophet", 90),
        ("TiDE", 45),
        ("TSMixer", 100),
    ],
)
def test_full_app_flow(app, model_choice, forecast_horizon, mocker):
    mock_st = mocker.patch("backend.app.streamlit.st")

    # Mock TimeSeries
    mock_time_series = mocker.Mock(spec=TimeSeries)
    mock_time_series.start_time.return_value = pd.Timestamp("2023-01-01")
    mock_time_series.end_time.return_value = pd.Timestamp("2023-12-31")
    mock_time_series.freq = "D"
    mock_time_series.__len__ = mocker.Mock(return_value=365)

    # Create a mock pandas DataFrame
    date_range = pd.date_range(start="2023-01-01", periods=365, freq="D")
    mock_df = pd.DataFrame(
        {
            "date": date_range,
            "value": np.random.rand(365),
        },
    ).set_index("date")
    mock_time_series.pd_dataframe.return_value = mock_df

    # Create complete mock session state
    mock_session_state = {
        "model_choice": model_choice,
        "forecast_horizon": forecast_horizon,
        "data": mock_time_series,
        "train_data": mock_time_series,
        "test_data": mock_time_series,
        "trained_models": {},
        "forecasts": {},
        "is_trained": False,
        "is_forecast_generated": False,
        "forecast_button": False,
    }

    # Set up the mock session state
    mock_st.session_state = mock_session_state

    # Important: Update the app's session state directly
    app.session_state = mock_session_state

    # Set up other mocks
    mock_load_data = mocker.patch.object(app.data_handler, "load_and_process_data")
    mock_load_data.return_value = (mock_time_series, mock_time_series, mock_time_series)

    mock_train_models = mocker.patch.object(app, "train_models")
    mock_generate_forecast = mocker.patch.object(app, "generate_forecast")
    mock_display_results = mocker.patch.object(app, "display_results")

    # Run the app
    app.run()

    # Assertions
    mock_st.title.assert_called_once_with("Time Series Forecasting App")
    mock_st.subheader.assert_called_once_with("Time Series Data")
    assert mock_st.line_chart.call_count == 1
    called_df = mock_st.line_chart.call_args[0][0]
    pd.testing.assert_frame_equal(called_df, mock_df)
    mock_train_models.assert_called_once_with(model_choice, "small")
    mock_generate_forecast.assert_called_once_with(forecast_horizon)
    mock_display_results.assert_called_once_with(model_choice, forecast_horizon)
