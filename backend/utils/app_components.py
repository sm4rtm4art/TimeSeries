"""
App components
"""
import traceback
from typing import Any, Dict, Union, Tuple

import pandas as pd
import streamlit as st
from darts import TimeSeries
from darts.metrics import mape, rmse, mae

from backend.models.chronos_model import ChronosPredictor
from backend.models.nbeats_model import NBEATSPredictor
from backend.models.prophet_model import ProphetModel
from backend.models.tide_model import TiDEPredictor
from backend.models.TFT_model import TFTPredictor
from backend.utils.plotting import TimeSeriesPlotter

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

def train_models(train_data: TimeSeries, test_data: TimeSeries, model_choice: str, model_size: str = "small") -> Dict[str, Any]:
    trained_models = {}
    models_to_train = ["N-BEATS", "Prophet", "TiDE", "Chronos"] if model_choice == "All Models" else [model_choice]
    
    for model in models_to_train:
        try:
            with st.spinner(f"Training {model} model... This may take a while"):
                if model == "Prophet":
                    current_model = ProphetModel()
                elif model == "N-BEATS":
                    current_model = NBEATSPredictor()
                elif model == "TiDE":
                    current_model = TiDEPredictor()
                elif model == "Chronos":
                    current_model = ChronosPredictor()
                else:
                    raise ValueError(f"Unknown model: {model}")
                
                current_model.train(train_data)
                trained_models[model] = current_model
                print(f"Successfully trained and added {model} to trained_models")

            st.success(f"{model} model trained successfully!")
        except Exception as e:
            st.error(f"Error training {model} model: {str(e)}")
            logger.error(f"Error training {model} model: {str(e)}")
            logger.error(traceback.format_exc())
    print(f"Trained models: {list(trained_models.keys())}")
    
    return trained_models

def fallback_historical_forecasts(model, series, start, forecast_horizon, stride=1, retrain=False):
    historical_forecasts = []
    for i in range(start, len(series) - forecast_horizon + 1, stride):
        train_data = series[:i]
        if retrain:
            model.train(train_data)
        forecast = model.predict(horizon=forecast_horizon)
        historical_forecasts.append(forecast)
    
    # Combine all forecasts into a single TimeSeries
    combined_forecast = TimeSeries.from_series(pd.concat([f.pd_series() for f in historical_forecasts]))
    return combined_forecast

def generate_forecasts(trained_models, data: TimeSeries, forecast_horizon: int, backtests: Dict[str, Dict[str, Union[TimeSeries, Dict[str, float]]]]) -> Dict[str, Dict[str, TimeSeries]]:
    forecasts = {}
    
    if not isinstance(trained_models, dict):
        logger.error(f"Expected trained_models to be a dictionary, but got {type(trained_models)}")
        return forecasts

    for model_name, model in trained_models.items():
        print(f"Attempting to generate forecast for {model_name}")
        try:
            # Generate the forecast
            logger.info(f"Generating forecast for {model_name}")
            if model_name == "Chronos":
                future_forecast = model.predict()
                if len(future_forecast) > forecast_horizon:
                    future_forecast = future_forecast[:forecast_horizon]
                elif len(future_forecast) < forecast_horizon:
                    logger.warning(f"Chronos forecast shorter than requested horizon. Padding with last value.")
                    last_value = future_forecast.values()[-1]
                    padding = [last_value] * (forecast_horizon - len(future_forecast))
                    future_forecast = TimeSeries.from_values(np.concatenate([future_forecast.values().flatten(), padding]))
            else:
                future_forecast = model.predict(horizon=forecast_horizon)
            
            # Generate future dates for the forecast
            future_dates = pd.date_range(start=data.end_time() + get_timedelta(data, 1), periods=forecast_horizon, freq=data.freq_str)
            logger.info(f"Generated future dates for {model_name}: {future_dates}")
            
            # Ensure the forecast has the correct time index
            if len(future_forecast) == len(future_dates):
                future_forecast = TimeSeries.from_times_and_values(future_dates, future_forecast.values())
            else:
                logger.warning(f"Forecast length ({len(future_forecast)}) doesn't match expected length ({len(future_dates)}). Using original forecast.")
            
            forecasts[model_name] = {
                'future': future_forecast,
                'backtest': backtests[model_name]['backtest'] if model_name in backtests else None
            }
            logger.info(f"Generated forecast for {model_name}: {future_forecast}")
            print(f"Successfully generated forecast for {model_name}")
        except Exception as e:
            logger.error(f"Error generating forecast for {model_name}: {type(e).__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            print(f"Error generating forecast for {model_name}: {str(e)}")
    print(f"Generated forecasts for: {list(forecasts.keys())}")
    return forecasts

def display_results(data: TimeSeries, train_data: TimeSeries, test_data: TimeSeries, 
                    forecasts: Dict[str, Dict[str, TimeSeries]], model_choice: str, forecast_horizon: int):
    plotter = TimeSeriesPlotter()

    st.header("Original Data")
    fig_original = plotter.plot_original_data(data)
    st.plotly_chart(fig_original, use_container_width=True)

    st.header("Train/Test Split with Backtest")
    fig_backtest = plotter.plot_train_test_with_backtest(train_data, test_data, forecasts, model_choice)
    st.plotly_chart(fig_backtest, use_container_width=True)
    
    st.header("Forecasting Results")
    fig_forecast = plotter.plot_forecasts(data, test_data, forecasts, model_choice)
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Display metrics
    st.header("Model Performance Metrics")
    
    logger.info(f"Calculating metrics for actual data of length {len(test_data)}")
    metrics = calculate_metrics_for_all_models(test_data, forecasts)
    
    if not metrics:
        st.warning("No metrics could be calculated. This might be due to misaligned data or forecast periods.")
    else:
        # Convert None values to "N/A" for display and handle potential errors
        formatted_metrics = {}
        for model, model_metrics in metrics.items():
            formatted_metrics[model] = {}
            for metric, value in model_metrics.items():
                if value is None:
                    formatted_metrics[model][metric] = "N/A"
                elif isinstance(value, str):  # This could be an error message
                    formatted_metrics[model][metric] = value
                else:
                    formatted_metrics[model][metric] = f"{value:.4f}"
        
        st.table(formatted_metrics)

    # Add annotation for the forecast horizon
    st.write(f"Forecast Horizon: {forecast_horizon} periods")

def perform_backtesting(trained_models, data: TimeSeries, test_data: TimeSeries) -> Dict[str, Dict[str, Union[TimeSeries, Dict[str, float]]]]:
    backtests = {}
    
    if not isinstance(trained_models, dict):
        logger.error(f"Expected trained_models to be a dictionary, but got {type(trained_models)}")
        return backtests

    for model_name, model in trained_models.items():
        print(f"Attempting to backtest {model_name}")
        try:
            if hasattr(model, 'backtest'):
                print(f"Calling backtest for {model_name}")
                backtest_start = len(data) - len(test_data)
                historical_forecasts, metrics = model.backtest(data, forecast_horizon=len(test_data), start=backtest_start)
                
                print(f"Backtest length: {len(historical_forecasts)}, Test data length: {len(test_data)}")
                
                # Ensure the historical forecasts match the test data length
                if len(historical_forecasts) != len(test_data):
                    logger.warning(f"Backtest length ({len(historical_forecasts)}) doesn't match test data length ({len(test_data)}). Adjusting...")
                    if len(historical_forecasts) > len(test_data):
                        historical_forecasts = historical_forecasts.slice(test_data.start_time(), test_data.end_time())
                    else:
                        # If backtest is shorter, pad it with NaN values
                        pad_length = len(test_data) - len(historical_forecasts)
                        pad_values = np.full((pad_length, historical_forecasts.width), np.nan)
                        pad_index = pd.date_range(start=historical_forecasts.end_time() + historical_forecasts.freq, periods=pad_length, freq=historical_forecasts.freq)
                        pad_series = TimeSeries.from_times_and_values(pad_index, pad_values)
                        historical_forecasts = historical_forecasts.append(pad_series)
                
                backtests[model_name] = {'backtest': historical_forecasts, 'metrics': metrics}
            else:
                logger.warning(f"{model_name} does not have a backtest method. Skipping backtesting.")
            print(f"Successfully completed backtesting for {model_name}")
        except Exception as e:
            print(f"Error during backtesting for {model_name}: {str(e)}")
            logger.error(traceback.format_exc())
    print(f"Completed backtests: {list(backtests.keys())}")
    return backtests

def calculate_metrics_for_all_models(actual: TimeSeries, forecasts: Dict[str, Dict[str, TimeSeries]]) -> Dict[str, Dict[str, Union[float, str]]]:
    metrics = {}
    for model_name, forecast_dict in forecasts.items():
        if 'backtest' in forecast_dict and forecast_dict['backtest'] is not None:
            backtest = forecast_dict['backtest']
            try:
                if len(backtest) == 0:
                    raise ValueError("Backtest is empty")
                if len(backtest) != len(actual):
                    raise ValueError(f"Backtest length ({len(backtest)}) doesn't match actual data length ({len(actual)})")
                
                metrics[model_name] = {
                    'MAPE': mape(actual, backtest),
                    'RMSE': rmse(actual, backtest),
                    'MAE': mae(actual, backtest)
                }
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                logger.error(f"Error calculating metrics for {model_name}: {error_msg}")
                metrics[model_name] = {
                    'MAPE': error_msg,
                    'RMSE': error_msg,
                    'MAE': error_msg
                }
        else:
            logger.warning(f"No backtest available for {model_name}. Skipping metric calculation.")
            metrics[model_name] = {
                'MAPE': "No backtest",
                'RMSE': "No backtest",
                'MAE': "No backtest"
            }
    return metrics