"""
App components
"""
from typing import Any, Dict

import pandas as pd
import numpy as np
import streamlit as st
from darts import TimeSeries
import traceback


from backend.models.chronos_model import ChronosPredictor, make_chronos_forecast
from backend.models.nbeats_model import NBEATSPredictor, make_nbeats_forecast
from backend.models.prophet_model import ProphetModel, make_prophet_forecast
from backend.models.tide_model import make_tide_forecast, train_tide_model
from backend.utils.metrics import calculate_metrics
from backend.utils.plotting import plot_all_forecasts, plot_forecast, plot_train_test_forecasts, plot_all_forecasts_without_test

def train_models(train_data: TimeSeries, model_choice: str, model_size: str = "small") -> Dict[str, Any]:
    trained_models = {}
    models_to_train = ["N-BEATS", "Prophet", "TiDE", "Chronos"] if model_choice == "All Models" else [model_choice]

    for model in models_to_train:
        with st.spinner(f"Training {model} model... This may take a few minutes."):
            try:
                if model == "N-BEATS":
                    nbeats_model = NBEATSPredictor()
                    nbeats_model.train(train_data)
                    trained_models[model] = nbeats_model
                elif model == "Prophet":
                    prophet_model = ProphetModel()
                    prophet_model.train(train_data.pd_dataframe())
                    trained_models[model] = prophet_model
                elif model == "TiDE":
                    tide_model, scaler = train_tide_model(train_data)
                    trained_models[model] = (tide_model, scaler)  # Store as tuple
                elif model == "Chronos":
                    chronos_model = ChronosPredictor(model_size)
                    chronos_model.train(train_data)
                    trained_models[model] = chronos_model
                st.success(f"{model} model trained successfully!")
            except Exception as e:
                error_msg = f"Error training {model} model: {type(e).__name__}: {str(e)}"
                print(error_msg)
                print("Traceback:")
                traceback.print_exc()
                st.error(error_msg)

    return trained_models

def generate_forecasts(trained_models: Dict[str, Any], data: TimeSeries, forecast_horizon: int) -> Dict[str, TimeSeries]:
    forecasts = {}
    for model_name, model in trained_models.items():
        try:
            if model_name == "Chronos":
                forecast = make_chronos_forecast(model, data, forecast_horizon)
            elif model_name == "N-BEATS":
                forecast = make_nbeats_forecast(model, data, forecast_horizon)
            elif model_name == "Prophet":
                forecast = make_prophet_forecast(model, forecast_horizon)
            elif model_name == "TiDE":
                tide_model, scaler = model  # Unpack the tuple
                forecast = make_tide_forecast(tide_model, scaler, data, forecast_horizon)
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            if isinstance(forecast, TimeSeries):
                forecasts[model_name] = forecast
            else:
                print(f"Forecast for {model_name} is not a TimeSeries object.")
                st.error(f"Forecast for {model_name} is not a TimeSeries object.")
        except Exception as e:
            error_msg = f"Error generating forecast for {model_name}: {type(e).__name__}: {str(e)}"
            print(error_msg)
            print("Traceback:")
            traceback.print_exc()
            st.error(error_msg)
    return forecasts

def display_results(
    data: TimeSeries,
    forecasts: Dict[str, Dict[str, TimeSeries]],
    test_data: TimeSeries,
    model_choice: str,
    forecast_horizon: int
) -> None:
    st.subheader("Train/Test Split and Test Period Forecasts")
    plot_train_test_forecasts(data, test_data, {model: forecast['test'] for model, forecast in forecasts.items()}, model_choice)
    
    st.subheader("Forecast Metrics (Test Period)")
    metrics = {}
    
    for model, forecast_dict in forecasts.items():
        try:
            test_forecast = forecast_dict['test']
            print(f"Processing forecast for {model}")
            print(f"Forecast: Start time = {test_forecast.start_time()}, End time = {test_forecast.end_time()}, Length = {len(test_forecast)}")
            print(f"Test data: Start time = {test_data.start_time()}, End time = {test_data.end_time()}, Length = {len(test_data)}")
            
            # Check if there's an overlap between forecast and test data
            if test_forecast.end_time() < test_data.start_time() or test_forecast.start_time() > test_data.end_time():
                print(f"Warning: No overlap between forecast and test data for {model}")
                metrics[model] = {
                    "MAE": None,
                    "MSE": None,
                    "RMSE": None,
                    "MAPE": None,
                    "sMAPE": None
                }
            else:
                # Find the common time range
                common_start = max(test_data.start_time(), test_forecast.start_time())
                common_end = min(test_data.end_time(), test_forecast.end_time())
                
                test_forecast_slice = test_forecast.slice(common_start, common_end)
                test_data_slice = test_data.slice(common_start, common_end)
                
                print(f"Sliced forecast: Start time = {test_forecast_slice.start_time()}, End time = {test_forecast_slice.end_time()}, Length = {len(test_forecast_slice)}")
                print(f"Sliced test data: Start time = {test_data_slice.start_time()}, End time = {test_data_slice.end_time()}, Length = {len(test_data_slice)}")
                
                metrics[model] = calculate_metrics(test_data_slice, test_forecast_slice)
        except Exception as e:
            st.warning(f"Unable to calculate metrics for {model}: {str(e)}")
            print(f"Error calculating metrics for {model}: {str(e)}")
            traceback.print_exc()
            metrics[model] = {
                "MAE": None,
                "MSE": None,
                "RMSE": None,
                "MAPE": None,
                "sMAPE": None
            }

    # Convert metrics to a DataFrame for better display
    metrics_df = pd.DataFrame(metrics).T
    st.table(metrics_df)

    st.subheader("Future Forecast")
    plot_all_forecasts_without_test(data, {model: forecast['future'] for model, forecast in forecasts.items()})