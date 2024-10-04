"""
App components
"""
import traceback
from typing import Any, Dict, Union, Tuple

import pandas as pd
import streamlit as st
from darts import TimeSeries

from backend.models.chronos_model import ChronosPredictor
from backend.models.nbeats_model import NBEATSPredictor
from backend.models.prophet_model import ProphetModel
from backend.models.tide_model import TiDEPredictor
from backend.models.time_mixer import TSMixerPredictor
from backend.models.TFT_model import TFTPredictor
from backend.utils.metrics import calculate_metrics_for_all_models
from backend.utils.plotting import plot_train_test_forecasts

import logging
from backend.utils.scaling import scale_data, inverse_scale
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_timedelta(series: TimeSeries, periods: int) -> pd.Timedelta:
    if len(series) < 2:
        raise ValueError("Series must have at least two data points to determine frequency.")
    
    # Try to infer frequency
    freq = pd.infer_freq(series.time_index)
    
    if freq is None:
        # If frequency can't be inferred, calculate the average time difference
        time_diff = series.time_index[-1] - series.time_index[0]
        avg_diff = time_diff / (len(series) - 1)
        return avg_diff * periods
    
    # Handle different frequency types
    if freq in ['D', 'H', 'T', 'S']:
        # For day, hour, minute, second frequencies
        return pd.Timedelta(periods, freq)
    elif freq in ['M', 'MS']:
        # For month frequencies
        return pd.offsets.MonthEnd(periods)
    elif freq in ['Y', 'YS']:
        # For year frequencies
        return pd.offsets.YearEnd(periods)
    elif freq == 'W':
        # For week frequency
        return pd.Timedelta(weeks=periods)
    else:
        # For other frequencies, use the difference between first two timestamps
        time_diff = series.time_index[1] - series.time_index[0]
        return time_diff * periods

def calculate_backtest_start(train_data: TimeSeries, test_data: TimeSeries, input_chunk_length: int) -> pd.Timestamp:
    # Calculate the ideal backtest start (test_data length before the end of train_data)
    ideal_start = train_data.end_time() - get_timedelta(train_data, len(test_data))
    
    # Ensure we have at least input_chunk_length periods of data before the start
    min_start = train_data.start_time() + get_timedelta(train_data, input_chunk_length)
    
    # Choose the later of ideal_start and min_start
    backtest_start = max(ideal_start, min_start)
    
    # If backtest_start is still after train_data.end_time(), adjust it
    if backtest_start >= train_data.end_time():
        backtest_start = train_data.end_time() - get_timedelta(train_data, 1)
    
    return backtest_start

def train_models(train_data: TimeSeries, test_data: TimeSeries, model_choice: str, model_size: str = "small") -> Tuple[Dict[str, Any], Dict[str, Dict[str, TimeSeries]]]:
    trained_models = {}
    backtests = {}
    forecasts = {}
    models_to_train = ["N-BEATS", "Prophet", "TiDE", "Chronos", "TSMixer", "TFT"] if model_choice == "All Models" else [model_choice]
    
    for model in models_to_train:
        try:
            with st.spinner(f"Training {model} model... This may take a while"):
                if model == "N-BEATS":
                    current_model = NBEATSPredictor(input_chunk_length=24, output_chunk_length=12)
                elif model == "Prophet":
                    current_model = ProphetModel()
                elif model == "TiDE":
                    current_model = TiDEPredictor()
                elif model == "Chronos":
                    current_model = ChronosPredictor(model_size)
                elif model == "TSMixer":
                    current_model = TSMixerPredictor(input_chunk_length=24, output_chunk_length=12)
                elif model == "TFT":
                    current_model = TFTPredictor(input_chunk_length=24, output_chunk_length=12)
                else:
                    raise ValueError(f"Unknown model: {model}")
                # Train the model
                current_model.train(train_data)
                trained_models[model] = current_model
                # Perform backtesting
                backtest_start = calculate_backtest_start(train_data, test_data, getattr(current_model, 'input_chunk_length', 24))
                forecast_horizon = min(len(test_data), (train_data.end_time() - backtest_start).days)
                backtest = current_model.historical_forecasts(
                    series=train_data,
                    start=backtest_start,
                    forecast_horizon=forecast_horizon,
                    stride=1,
                    retrain=False,
                    verbose=True
                )
                backtests[model] = {'backtest': backtest}
                # Generate future forecast
                future_forecast = current_model.predict(horizon=len(test_data), data=train_data)
                forecasts[model] = {'future': future_forecast, 'backtest': backtest}
            
            print(f"Generating forecast for {model}")
        except Exception as e:
            error_msg = f"Error training {model} model: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            st.error(error_msg)
            forecasts[model] = {'future': None, 'backtest': None}
    
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


def display_results(data: TimeSeries, test_data: Union[TimeSeries, Dict[str, TimeSeries]], forecasts: Dict[str, Dict[str, TimeSeries]], model_choice: str, forecast_horizon: int):
    st.header("Forecasting Results")
    
    # Plot forecasts
    fig = plot_train_test_forecasts(data, test_data, forecasts, model_choice)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display metrics
    st.header("Model Performance Metrics")
    
    # Check if test_data is a TimeSeries or a dictionary
    if isinstance(test_data, TimeSeries):
        actual_data = test_data
    elif isinstance(test_data, dict) and 'future' in test_data:
        actual_data = test_data['future']
    else:
        st.error("Invalid test data format")
        logger.error(f"Invalid test data format: {type(test_data)}")
        return

    logger.info(f"Calculating metrics for actual data of length {len(actual_data)}")
    metrics = calculate_metrics_for_all_models(actual_data, forecasts)
    
    if not metrics:
        st.warning("No metrics could be calculated. This might be due to misaligned data or forecast periods.")
        return

    # Convert None values to "N/A" for display
    formatted_metrics = {
        model: {metric: "N/A" if value is None else f"{value:.4f}" for metric, value in model_metrics.items()}
        for model, model_metrics in metrics.items()
    }
    
    st.table(formatted_metrics)

    # Add annotation for the forecast horizon
    st.write(f"Forecast Horizon: {forecast_horizon} periods")