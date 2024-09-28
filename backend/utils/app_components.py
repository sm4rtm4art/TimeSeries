"""
App components
"""
from typing import Any, Dict

import pandas as pd
import streamlit as st
from darts import TimeSeries

from backend.models.chronos_model import make_chronos_forecast, train_chronos_model
from backend.models.nbeats_model import make_nbeats_forecast, train_nbeats_model
from backend.models.prophet_model import make_prophet_forecast, train_prophet_model
from backend.models.tide_model import make_tide_forecast, train_tide_model
from backend.utils.metrics import calculate_metrics
from backend.utils.plotting import plot_all_forecasts, plot_forecast

# Set up logging
#logging.basicConfig(level=logging.INFO)
#logger = logging.getLogger(__name__)

def train_models(train_data: TimeSeries, model_choice: str, model_size: str = "small") -> Dict[str, Any]:
    trained_models = {}
    models_to_train = ["N-BEATS", "Prophet", "TiDE", "Chronos"] if model_choice == "All Models" else [model_choice]

    for model in models_to_train:
        with st.spinner(f"Training {model} model... This may take a few minutes."):
            try:
                st.text(f"Starting {model} model training...")
                if model == "N-BEATS":
                    trained_models[model] = train_nbeats_model(train_data)
                elif model == "Prophet":
                    trained_models[model] = train_prophet_model(train_data)
                elif model == "TiDE":
                    trained_models[model] = train_tide_model(train_data)
                elif model == "Chronos":
                    trained_models[model] = train_chronos_model(train_data, model_size)
                st.success(f"{model} model trained successfully!")
            except Exception as e:
                st.error(f"Error training {model} model: {str(e)}")
                st.text(f"Error details: {type(e).__name__}: {str(e)}")

    return trained_models


def generate_forecasts(models: Dict[str, Any], data: TimeSeries, forecast_horizon: int) -> Dict[str, TimeSeries]:
    forecasts = {}
    for model_name, model in models.items():
        try:
            if model_name == "N-BEATS":
                forecast = make_nbeats_forecast(model, data, forecast_horizon)
            elif model_name == "Prophet":
                forecast = make_prophet_forecast(model, forecast_horizon)
            elif model_name == "TiDE":
                forecast, error_msg = make_tide_forecast(model, data, forecast_horizon)
                if error_msg:
                    st.error(f"Error generating forecast for {model_name}: {error_msg}")
                    continue
            elif model_name == "Chronos":
                forecast = make_chronos_forecast(model, data, forecast_horizon)
            
            if isinstance(forecast, TimeSeries):
                forecasts[model_name] = forecast
            else:
                st.error(f"Forecast for {model_name} is not a TimeSeries object.")
        except Exception as e:
            st.error(f"Error generating forecast for {model_name}: {str(e)}")
    return forecasts


def display_results(
    data: TimeSeries,
    forecasts: Dict[str, Any],
    test_data: TimeSeries,
    model_choice: str
) -> None:
    st.subheader("Train/Test Split")
    train_data = data.slice(data.start_time(), test_data.start_time() - data.freq)
    df_split = pd.concat([
        train_data.pd_dataframe().rename(columns={train_data.columns[0]: 'Training Data'}),
        test_data.pd_dataframe().rename(columns={test_data.columns[0]: 'Test Data'})
    ])
    st.line_chart(df_split)

    st.subheader("Forecast Results")
    if model_choice == "All Models":
        plot_all_forecasts(data, forecasts, test_data)
    else:
        if model_choice in forecasts:
            plot_forecast(data, forecasts[model_choice], model_choice, test_data)
        else:
            st.error(f"No forecast available for {model_choice}")
            return

    st.subheader("Forecast Metrics (Test Period)")
    metrics = {}
    for model, forecast in forecasts.items():
        if model == "Chronos":
            st.info("Metrics for Chronos model are not available due to its unique output format.")
            continue

        if not isinstance(test_data, TimeSeries):
            st.warning(f"Test data is not a TimeSeries object. Skipping metrics calculation for {model}.")
            continue

        try:
            # Ensure forecast and test_data have the same length
            test_forecast = forecast.slice(test_data.start_time(), test_data.end_time())
            if len(test_forecast) != len(test_data):
                st.warning(f"Forecast length for {model} doesn't match test data length. Adjusting...")
                if len(test_forecast) > len(test_data):
                    test_forecast = test_forecast[:len(test_data)]
                else:
                    # Extend the forecast to match the test data length
                    additional_length = len(test_data) - len(test_forecast)
                    last_value = test_forecast.values()[-1][0]
                    additional_values = [last_value] * additional_length
                    additional_times = [test_forecast.end_time() + i * test_forecast.freq for i in range(1, additional_length + 1)]
                    additional_series = TimeSeries.from_times_and_values(additional_times, additional_values)
                    test_forecast = test_forecast.append(additional_series)

            metrics[model] = calculate_metrics(test_data, test_forecast)
        except Exception as e:
            st.warning(f"Unable to calculate metrics for {model}: {str(e)}")
            metrics[model] = {"Error": str(e)}

    st.write(metrics)


def display_results_without_test(
    data: TimeSeries,
    forecasts: Dict[str, Any],
    model_choice: str
) -> None:
    st.subheader("Forecast Results")
    if model_choice == "All Models":
        plot_all_forecasts_without_test(data, forecasts)
    else:
        if model_choice in forecasts:
            plot_forecast(data, forecasts[model_choice], model_choice)
        else:
            st.error(f"No forecast available for {model_choice}")
            return

    st.info("Test data is not available, so metrics cannot be calculated.")

