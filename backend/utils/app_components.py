"""
App components
"""
from typing import Any, Dict

import pandas as pd
import streamlit as st
from darts import TimeSeries
import numpy as np
import traceback

from backend.models.chronos_model import ChronosPredictor, make_chronos_forecast
from backend.models.nbeats_model import NBEATSPredictor, make_nbeats_forecast
from backend.models.prophet_model import ProphetModel, make_prophet_forecast
from backend.models.tide_model import TiDEPredictor, make_tide_forecast
from backend.utils.metrics import calculate_metrics
from backend.utils.plotting import plot_all_forecasts, plot_forecast, plot_train_test_forecasts, plot_all_forecasts_without_test, plot_forecasts
from backend.utils.tensor_utils import ensure_float32, is_mps_available
from backend.utils.scaling import scale_data, inverse_scale_forecast

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
                    tide_model = TiDEPredictor()
                    tide_model.train(train_data)
                    trained_models[model] = tide_model
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

def generate_forecasts(trained_models: Dict[str, Any], data: TimeSeries, test_data: TimeSeries, forecast_horizon: int) -> Dict[str, Dict[str, TimeSeries]]:
    forecasts = {}
    
    for model_name, model in trained_models.items():
        try:
            print(f"Generating forecast for {model_name}")
            backtest_start = len(data) - len(test_data)
            
            if model_name == "N-BEATS":
                print("Generating N-BEATS backtest")
                backtest_forecast = model.backtest(data, start=backtest_start, forecast_horizon=len(test_data))
                print("N-BEATS backtest generated")
                print("Generating N-BEATS future forecast")
                future_forecast = make_nbeats_forecast(model, data, forecast_horizon)
                print("N-BEATS future forecast generated")
            elif model_name == "Prophet":
                print("Generating Prophet backtest")
                backtest_forecast = model.backtest(data.pd_dataframe(), periods=len(test_data))
                print("Prophet backtest generated")
                print("Generating Prophet future forecast")
                future_forecast = make_prophet_forecast(model, forecast_horizon)
                print("Prophet future forecast generated")
            elif model_name == "TiDE":
                print("Generating TiDE backtest")
                backtest_forecast = model.backtest(data, start=backtest_start, forecast_horizon=len(test_data))
                print("TiDE backtest generated")
                print("Generating TiDE future forecast")
                future_forecast = make_tide_forecast(model, data, forecast_horizon)
                print("TiDE future forecast generated")
            elif model_name == "Chronos":
                print("Generating Chronos backtest")
                backtest_forecast = model.backtest(data, start=backtest_start, forecast_horizon=len(test_data))
                print("Chronos backtest generated")
                print("Generating Chronos future forecast")
                future_forecast = make_chronos_forecast(model, data, forecast_horizon)
                print("Chronos future forecast generated")
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            print(f"Backtest forecast shape: {backtest_forecast.shape}")
            print(f"Future forecast shape: {future_forecast.shape}")
            
            forecasts[model_name] = {
                'backtest': backtest_forecast,
                'future': future_forecast
            }
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
    st.subheader("Train/Test Split with Backtesting")
    plot_train_test_forecasts(data, test_data, forecasts, model_choice)

    st.subheader("Forecast Metrics (Test Period)")
    metrics = {}
    
    for model, forecast_dict in forecasts.items():
        try:
            metrics[model] = calculate_metrics(test_data, forecast_dict['backtest'])
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

    st.subheader("Future Forecasts")
    plot_forecasts(data, test_data, forecasts, model_choice)