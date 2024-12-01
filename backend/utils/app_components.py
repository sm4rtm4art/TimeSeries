"""
App components for Time Series Forecasting Application
"""
import traceback
from typing import Any, Dict, Union, Tuple, Optional

import pandas as pd
import streamlit as st
from darts import TimeSeries
from darts.metrics import mape, rmse, mae, mse

from backend.core.interfaces.base_model import TimeSeriesPredictor
from backend.core.model_factory import ModelFactory
from backend.core.trainer import ModelTrainer
from backend.core.evaluator import ModelEvaluator
from backend.utils.plotting import TimeSeriesPlotter
from backend.utils.scaling import scale_data, inverse_scale
from backend.utils.time_utils import TimeSeriesUtils

from backend.domain.models.deep_learning.nbeats import NBEATSPredictor
from backend.domain.models.deep_learning.time_mixer import TSMixerPredictor

import logging
logger = logging.getLogger(__name__)

class ForecastingService:
    @staticmethod
    def train_models(train_data: TimeSeries, test_data: TimeSeries, model_choice: str, model_size: str = "small"):
        """Train selected models and perform backtesting."""
        try:
            return ModelTrainer.train_models(
                train_data=train_data,
                model_choice=model_choice,
                model_size=model_size
            )
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    @staticmethod
    def generate_forecasts(trained_models, data: TimeSeries, forecast_horizon: int, 
                         backtests: Dict[str, Dict[str, Union[TimeSeries, Dict[str, float]]]]
    ) -> Dict[str, Dict[str, TimeSeries]]:
        """Generate forecasts for all trained models."""
        forecasts = {}
        
        if not isinstance(trained_models, dict):
            logger.error(f"Expected trained_models to be a dictionary, but got {type(trained_models)}")
            return forecasts

        for model_name, model in trained_models.items():
            print(f"Attempting to generate forecast for {model_name}")
            try:
                # Generate the forecast using n instead of horizon
                logger.info(f"Generating forecast for {model_name}")
                future_forecast = model.predict(n=forecast_horizon)
                
                # Generate future dates for the forecast
                future_dates = pd.date_range(
                    start=data.end_time() + TimeSeriesUtils.get_timedelta(data, 1),
                    periods=forecast_horizon,
                    freq=data.freq_str
                )
                
                if len(future_forecast) == len(future_dates):
                    future_forecast = TimeSeries.from_times_and_values(
                        future_dates,
                        future_forecast.values()
                    )
                
                forecasts[model_name] = {
                    'future': future_forecast,
                    'backtest': backtests[model_name]['backtest'] if model_name in backtests else None,
                    'metrics': backtests[model_name]['metrics'] if model_name in backtests else None
                }
                
                logger.info(f"Generated forecast for {model_name}")
                print(f"Successfully generated forecast for {model_name}")
            except Exception as e:
                logger.error(f"Error generating forecast for {model_name}: {str(e)}")
                logger.error(traceback.format_exc())
                continue
                
        print(f"Generated forecasts for: {list(forecasts.keys())}")
        return forecasts

    @staticmethod
    def perform_backtesting(data: TimeSeries, test_data: TimeSeries, trained_models: Dict[str, TimeSeriesPredictor], horizon: int, stride: int = 1) -> Dict[str, Dict[str, Union[TimeSeries, Dict[str, float]]]]:
        """Perform backtesting for all models."""
        backtests = {}
        
        for model_name, model in trained_models.items():
            try:
                logger.info(f"\nBacktesting {model_name}...")
                backtest_result = model.backtest(
                    data=data,
                    start=0.6,  # Use last 40% of data
                    forecast_horizon=horizon,
                    stride=stride
                )
                backtests[model_name] = backtest_result
                
            except Exception as e:
                logger.error(f"Error in backtesting {model_name}: {str(e)}")
                logger.error(traceback.format_exc())
                continue
                
        return backtests

    @staticmethod
    def calculate_metrics_for_all_models(actual: TimeSeries, forecasts: Dict[str, Dict[str, TimeSeries]]) -> Dict[str, Dict[str, Union[float, str]]]:
        metrics = {}
        
        for model_name, forecast_dict in forecasts.items():
            logger.info(f"Calculating metrics for {model_name}")
            
            try:
                if 'backtest' not in forecast_dict or forecast_dict['backtest'] is None:
                    raise ValueError("No backtest available")
                    
                backtest = forecast_dict['backtest']
                
                # Ensure the time indices match
                if not actual.time_index.equals(backtest.time_index):
                    logger.warning(f"Time indices don't match for {model_name}. Attempting to align...")
                    common_indices = actual.time_index.intersection(backtest.time_index)
                    if len(common_indices) == 0:
                        raise ValueError("No common time indices between actual and backtest data")
                        
                    actual_aligned = actual.slice(common_indices[0], common_indices[-1])
                    backtest_aligned = backtest.slice(common_indices[0], common_indices[-1])
                else:
                    actual_aligned = actual
                    backtest_aligned = backtest
                
                logger.info(f"Computing metrics between series of lengths: actual={len(actual_aligned)}, backtest={len(backtest_aligned)}")
                
                metrics[model_name] = {
                    'MAPE': float(mape(actual_aligned, backtest_aligned)),
                    'RMSE': float(rmse(actual_aligned, backtest_aligned)),
                    'MAE': float(mae(actual_aligned, backtest_aligned))
                }
                
                logger.info(f"Successfully calculated metrics for {model_name}: {metrics[model_name]}")
            except Exception as e:
                logger.error(f"Error calculating metrics for {model_name}: {str(e)}")
                logger.error(traceback.format_exc())
                continue
        return metrics

    @staticmethod
    def debug_data_state(data: TimeSeries, test_data: TimeSeries, historical_forecasts: TimeSeries = None):
        """Helper function to debug data alignment issues"""
        print("=== Data State Debug ===")
        print(f"Full data range: {data.start_time()} to {data.end_time()}")
        print(f"Test data range: {test_data.start_time()} to {test_data.end_time()}")
        if historical_forecasts is not None:
            print(f"Historical forecasts range: {historical_forecasts.start_time()} to {historical_forecasts.end_time()}")
        print(f"Full data length: {len(data)}")
        print(f"Test data length: {len(test_data)}")
        if historical_forecasts is not None:
            print(f"Historical forecasts length: {len(historical_forecasts)}")
        print("=====================")

    @staticmethod
    def display_metrics(model_metrics: Dict[str, Dict[str, float]]):
        """Display metrics for each model in a formatted way."""
        try:
            # Create a DataFrame to store all metrics
            metrics_data = []
            
            for model_name, metrics_dict in model_metrics.items():
                if isinstance(metrics_dict, dict):
                    model_data = {'Model': model_name}
                    for metric_name, value in metrics_dict.items():
                        if isinstance(value, (int, float)):
                            model_data[metric_name] = f"{value:.4f}"
                        else:
                            model_data[metric_name] = str(value)
                    metrics_data.append(model_data)
            
            if metrics_data:
                # Convert to DataFrame and display as table
                metrics_df = pd.DataFrame(metrics_data)
                st.write("Model Performance Metrics:")
                st.table(metrics_df.set_index('Model'))
            else:
                st.warning("No metrics available")
                
        except Exception as e:
            logger.error(f"Error displaying metrics: {str(e)}")
            st.error("Error displaying metrics")

    @staticmethod
    def display_forecasts(data: TimeSeries, forecasts: Dict[str, Dict[str, TimeSeries]], model_choice: str) -> None:
        """Display forecasts for each model."""
        try:
            plotter = TimeSeriesPlotter()
            
            if model_choice == "All Models":
                # Plot all models
                for model_name, forecast_dict in forecasts.items():
                    plotter.plot_forecast(data, forecast_dict['future'], model_name)
        except Exception as e:
            logger.error(f"Error in display_forecasts: {str(e)}")
            logger.error(traceback.format_exc())
            st.error(f"Error displaying forecasts: {str(e)}")