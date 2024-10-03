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

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_models(train_data: TimeSeries, test_data: TimeSeries, model_choice: str, model_size: str = "small") -> Dict[str, Any]:
    trained_models = {}
    backtests = {}
    models_to_train = ["N-BEATS", "Prophet", "TiDE", "Chronos"] if model_choice == "All Models" else [model_choice]

    # Combine train_data and test_data to get the full dataset
    full_data = train_data.append(test_data)
    print(f"Full data length: {len(full_data)}, Train data length: {len(train_data)}, Test data length: {len(test_data)}")

    for model in models_to_train:
        with st.spinner(f"Training {model} model... This may take a few minutes."):
            try:
                if model == "N-BEATS":
                    nbeats_model = NBEATSPredictor()
                    nbeats_model.train(train_data)
                    trained_models[model] = nbeats_model
                elif model == "Prophet":
                    prophet_model = ProphetModel()
                    prophet_model.train(train_data)
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
                backtest_start = len(train_data) - len(test_data)
                forecast_horizon = len(test_data)
                
                print(f"Backtesting {model}. Start: {backtest_start}, Horizon: {forecast_horizon}")
                try:
                    backtest = trained_models[model].backtest(data=full_data, start=backtest_start, forecast_horizon=forecast_horizon)
                    print(f"Backtest result for {model}: Length = {len(backtest)}")
                    backtests[model] = backtest
                except Exception as backtest_error:
                    print(f"Error during backtesting for {model}: {str(backtest_error)}")
                    backtests[model] = None

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
    print(f"Generating forecasts. Full data length: {len(data)}, Test data length: {len(test_data)}, Forecast horizon: {forecast_horizon}")
    
    for model_name, model in trained_models.items():
        try:
            print(f"Generating forecast for {model_name}")
            if forecast_horizon <= 0:
                raise ValueError(f"Invalid forecast horizon: {forecast_horizon}")
            
            # Ensure forecast_horizon is an integer
            horizon = int(forecast_horizon)
            
            if model_name == "TiDE":
                future_forecast = model.predict(horizon=horizon)
            else:
                future_forecast = model.predict(horizon=horizon, data=data)
            
            backtest = backtests.get(model_name)
            
            if future_forecast is None:
                raise ValueError(f"Null forecast generated for {model_name}")
            
            print(f"Generated forecast for {model_name}: Future Length = {len(future_forecast)}, Backtest Length = {len(backtest) if backtest else 'N/A'}")
            
            forecasts[model_name] = {
                'future': future_forecast,
                'backtest': backtest
            }
        except Exception as e:
            error_msg = f"Error generating forecast for {model_name}: {type(e).__name__}: {str(e)}"
            print(error_msg)
            print("Traceback:")
            import traceback
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