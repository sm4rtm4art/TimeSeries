"""
App components
"""
from typing import Any, Dict

import pandas as pd
import numpy as np

import streamlit as st
from darts import TimeSeries
from models.chronos_model import ChronosPredictor, make_chronos_forecast
from models.nbeats_model import NBEATSPredictor, make_nbeats_forecast
from models.prophet_model import ProphetModel, make_prophet_forecast
from models.tide_model import make_tide_forecast, train_tide_model
from utils.metrics import calculate_metrics
from utils.plotting import plot_all_forecasts, plot_forecast


def train_models(train_data: Any,
                model_choice: str,
                model_size: str = "small") -> Dict[str, Any]:
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
                    trained_models[model] = (tide_model, scaler)
                elif model == "Chronos":
                    chronos_model = ChronosPredictor(model_size)
                    chronos_model.train(train_data)
                    trained_models[model] = chronos_model
                st.success(f"{model} model trained successfully!")
            except Exception as e:
                st.error(f"Error training {model} model: {str(e)}")

    return trained_models


def generate_forecasts(trained_models: Dict[str, Any],
                       train_data: TimeSeries,
                       test_data_length: int,
                       forecast_horizon: int) -> Dict[str, TimeSeries]:
    forecasts = {}
    for model_name, model in trained_models.items():
        if model_name == "Chronos":
            forecast = make_chronos_forecast(model, train_data, forecast_horizon)
        elif model_name == "N-BEATS":
            forecast = make_nbeats_forecast(model, forecast_horizon)
        elif model_name == "Prophet":
            forecast = make_prophet_forecast(model, forecast_horizon)
        elif model_name == "TiDE":
            tide_model, scaler = model
            forecast = make_tide_forecast(tide_model, scaler, forecast_horizon)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Ensure forecast is a single TimeSeries object
        if not isinstance(forecast, TimeSeries):
            raise TypeError(f"Forecast for {model_name} is not a TimeSeries object")
        
        forecasts[model_name] = forecast

    return forecasts


def display_results(
    data: Any,
    forecasts: Dict[str, Any],
    test_data: Any,
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
        test_forecast = forecast.slice(test_data.start_time(), test_data.end_time())
        metrics[model] = calculate_metrics(test_data, test_forecast)

    st.table(pd.DataFrame(metrics).T)
