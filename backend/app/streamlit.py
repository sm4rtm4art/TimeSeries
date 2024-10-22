"""
Time Series Forecasting Streamlit Application

This module serves as the main entry point for the Streamlit-based user interface
of the Time Series Forecasting application. It orchestrates data loading, model training,
forecasting, and result visualization through an interactive web interface.

The application allows users to:
1. Load and visualize time series data
2. Train various forecasting models
3. Generate and visualize forecasts
4. Compare model performances

Classes:
    DataHandler: Handles data loading and processing.
    TimeSeriesForecastApp: Main application class that coordinates the UI and other components.

Usage:
    Run this script directly to start the Streamlit application:
    $ streamlit run backend/app/streamlit.py
"""

import streamlit as st
import logging
from typing import Tuple, Dict, Any
from darts import TimeSeries

from backend.data.data_loader import DataLoader
from backend.utils.app_components import (
    train_models, 
    generate_forecasts, 
    perform_backtesting, 
    display_results
)
from backend.utils.session_state import initialize_session_state, get_session_state
from backend.utils.ui_components import display_sidebar, display_data_info
from backend.utils.plotting import TimeSeriesPlotter
import traceback

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class DataHandlingError(Exception):
    """Custom exception for data handling errors."""
    pass

class ModelTrainingError(Exception):
    """Custom exception for model training errors."""
    pass

class ForecastingError(Exception):
    """Custom exception for forecasting errors."""
    pass

class DataHandler:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def load_and_process_data(self) -> Tuple[TimeSeries, TimeSeries, TimeSeries]:
        try:
            data = self.data_loader.load_data()
            train_data = data[:int(0.8 * len(data))]
            test_data = data[int(0.8 * len(data)):]
            return data, train_data, test_data
        except Exception as e:
            raise DataHandlingError(f"Error loading or processing data: {str(e)}")

class TimeSeriesForecastApp:
    def __init__(self, data_handler: DataHandler):
        self.data_handler = data_handler
        self.session_state = st.session_state

    def run(self):
        st.title("Time Series Forecasting App")
        
        if 'data' not in self.session_state:
            self.load_data()
        
        if self.session_state.get('data') is not None:
            self.display_data()
            # Use the model_choice directly from session_state without a default value
            model_choice = self.session_state['model_choice']
            forecast_horizon = self.session_state['forecast_horizon']
            if self.train_models(model_choice, "small"):
                self.generate_forecast(forecast_horizon)
                self.display_results(model_choice, forecast_horizon)

    def load_data(self):
        try:
            data, train_data, test_data = self.data_handler.load_and_process_data()
            self.session_state['data'] = data
            self.session_state['train_data'] = train_data
            self.session_state['test_data'] = test_data
        except Exception as e:
            raise DataHandlingError(f"Error loading data: {str(e)}")

    def display_data(self):
        st.subheader("Time Series Data")
        st.line_chart(self.session_state['data'].pd_dataframe())

    def train_models(self, model_choice, model_size) -> bool:
        try:
            train_models(
                self.session_state.train_data,
                self.session_state.test_data,
                model_choice,
                model_size
            )
            return True
        except ModelTrainingError as e:
            st.error(f"An error occurred during model training: {str(e)}")
            return False

    def generate_forecast(self, forecast_horizon):
        try:
            generate_forecasts(
                self.session_state.trained_models,
                self.session_state.data,
                forecast_horizon,
                self.session_state.backtests
            )
        except ForecastingError as e:
            st.error(f"An error occurred during forecasting: {str(e)}")

    def display_results(self, model_choice, forecast_horizon):
        try:
            display_results(
                self.session_state.data,
                self.session_state.train_data,
                self.session_state.test_data,
                self.session_state.forecasts,
                model_choice,
                forecast_horizon,
            )
        except Exception as e:
            st.error(f"An error occurred while displaying results: {str(e)}")
            logger.exception("Error while displaying results")

def main():
    initialize_session_state()
    
    st.title("Time Series Forecasting App")
    
    model_choice, model_size, train_button, forecast_horizon, forecast_button = display_sidebar()
    
    data_loader = DataLoader()
    data, train_data, test_data = data_loader.load_data()
    
    if data is not None:
        st.session_state.data = data
        st.session_state.train_data = train_data
        st.session_state.test_data = test_data
        
        # Display original data
        st.subheader("Original Data")
        st.line_chart(data.pd_dataframe())

        # Display train/test split
        st.subheader("Train/Test Split")
        split_data = train_data.pd_dataframe()
        split_data["Test"] = test_data.pd_dataframe()
        st.line_chart(split_data)

        # Train models
        if train_button:
            with st.spinner("Training models... This may take a few minutes."):
                try:
                    st.session_state.trained_models = train_models(
                        st.session_state.train_data, st.session_state.test_data, model_choice, model_size
                    )
                    st.success("Models trained successfully!")
                except Exception as e:
                    st.error(f"An error occurred during model training: {str(e)}")
                    logger.error(f"An error occurred during model training: {str(e)}")
                    logger.error(traceback.format_exc())

        # Perform backtesting and generate forecasts
        if 'trained_models' in st.session_state and forecast_button:
            with st.spinner("Generating forecasts and performing backtesting..."):
                try:
                    backtests = perform_backtesting(
                        st.session_state.trained_models,
                        st.session_state.data,
                        st.session_state.test_data
                    )
                    forecasts = generate_forecasts(
                        st.session_state.trained_models,
                        st.session_state.data,
                        forecast_horizon,
                        backtests
                    )
                    
                    # Combine backtests and forecasts
                    for model_name in forecasts:
                        if model_name in backtests:
                            forecasts[model_name]['backtest'] = backtests[model_name]['backtest']
                    
                    st.session_state.is_forecast_generated = True
                    st.session_state.forecasts = forecasts
                    st.success("Forecasts generated and backtesting performed successfully!")
                except Exception as e:
                    st.error(f"An error occurred during forecast generation or backtesting: {str(e)}")
                    logger.error(f"An error occurred during forecast generation or backtesting: {str(e)}")
                    logger.error(traceback.format_exc())

        # Display results
        if 'is_forecast_generated' in st.session_state and st.session_state.is_forecast_generated:
            try:
                display_results(
                    st.session_state.data,
                    st.session_state.train_data,
                    st.session_state.test_data,
                    st.session_state.forecasts,
                    model_choice,
                    forecast_horizon,
                )
            except Exception as e:
                st.error(f"An error occurred while displaying results: {str(e)}")
                logger.error(f"An error occurred while displaying results: {str(e)}")
                logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
