"""
App components
"""
import traceback
from typing import Any, Dict

import pandas as pd
import streamlit as st
from darts import TimeSeries

from backend.models.chronos_model import ChronosPredictor
from backend.models.nbeats_model import NBEATSPredictor
from backend.models.prophet_model import ProphetModel
from backend.models.tide_model import TiDEPredictor
from backend.utils.metrics import calculate_metrics
from backend.utils.plotting import (
    plot_forecasts,
    plot_train_test_forecasts,
)


def train_models(train_data: TimeSeries, test_data: TimeSeries, model_choice: str, model_size: str = "small") -> Dict[str, Any]:
    trained_models = {}
    backtests = {}
    models_to_train = ["N-BEATS", "Prophet", "TiDE", "Chronos"] if model_choice == "All Models" else [model_choice]

    # Combine train_data and test_data to get the full dataset
    full_data = train_data.append(test_data)

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
                
                # Perform backtesting
                backtest_start = len(train_data)
                backtest = trained_models[model].backtest(full_data, start=backtest_start, forecast_horizon=len(test_data))
                backtests[model] = backtest

                st.success(f"{model} model trained and backtested successfully!")
            except Exception as e:
                error_msg = f"Error training {model} model: {type(e).__name__}: {str(e)}"
                print(error_msg)
                print("Traceback:")
                traceback.print_exc()
                st.error(error_msg)

    return trained_models, backtests


def generate_forecasts(trained_models: Dict[str, Any], data: TimeSeries, test_data: TimeSeries, forecast_horizon: int, backtests: Dict[str, TimeSeries]) -> Dict[str, Dict[str, TimeSeries]]:
    forecasts = {}

    for model_name, model in trained_models.items():
        try:
            print(f"Generating forecast for {model_name}")
            backtest_forecast = backtests[model_name]  # Use the pre-computed backtest
            future_forecast = model.predict(data, forecast_horizon)

            print(f"Backtest forecast: Length = {len(backtest_forecast)}, Dimensions = {backtest_forecast.n_components}")
            print(f"Future forecast: Length = {len(future_forecast)}, Dimensions = {future_forecast.n_components}")

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