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
from typing import Tuple
from darts import TimeSeries
import traceback
import ipdb

from backend.data.data_loader import DataLoader
from backend.core.trainer import ModelTrainer
from backend.utils.app_components import (
    generate_forecasts, 
    perform_backtesting, 
    display_results
)
from backend.utils.session_state import initialize_session_state
from backend.utils.ui_components import display_sidebar

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
            trained_models = ModelTrainer.train_models(
                self.session_state.train_data,
                model_choice,
                model_size
            )
            if trained_models:
                self.session_state.trained_models = trained_models
                return True
            return False
        except Exception as e:
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
            # 1. Log session state contents
            logger.info("\n=== Session State Contents ===")
            logger.info(f"Available keys: {self.session_state.__dict__.keys()}")
            
            # 2. Log backtests structure
            logger.info("\n=== Backtests Structure ===")
            if hasattr(self.session_state, 'backtests'):
                logger.info(f"Backtests keys: {self.session_state.backtests.keys()}")
                # Log first model's backtest structure
                first_model = list(self.session_state.backtests.keys())[0]
                logger.info(f"\nExample structure for {first_model}:")
                logger.info(f"Keys: {self.session_state.backtests[first_model].keys()}")
                logger.info(f"Metrics: {self.session_state.backtests[first_model].get('metrics', 'No metrics found')}")
            else:
                logger.info("No backtests in session state")
                
            # 3. Log model metrics
            logger.info("\n=== Model Metrics ===")
            if hasattr(self.session_state, 'model_metrics'):
                logger.info(f"Model metrics: {self.session_state.model_metrics}")
            else:
                logger.info("No model metrics in session state")
                
            # Now call display_results
            display_results(
                data=self.session_state.data,
                train_data=self.session_state.train_data,
                test_data=self.session_state.test_data,
                forecasts=self.session_state.forecasts,
                backtests=self.session_state.backtests,
                model_metrics=self.session_state.model_metrics,
                model_choice=model_choice
            )
            
        except Exception as e:
            logger.error("\n=== Error in display_results ===")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error message: {str(e)}")
            logger.error("Full traceback:", exc_info=True)
            st.error(f"An error occurred while displaying results: {str(e)}")


def main():
    try:
        # Initialize session state first
        initialize_session_state()
        
        st.title("Time Series Forecasting App")
        
        # Get UI components
        model_choice, model_size, train_button, forecast_horizon, forecast_button = display_sidebar()
        
        # Update session state with sidebar values
        st.session_state.model_choice = model_choice
        st.session_state.model_size = model_size
        st.session_state.train_button = train_button
        st.session_state.forecast_horizon = forecast_horizon
        st.session_state.forecast_button = forecast_button
        
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
            if st.session_state.train_button:
                try:
                    print("=== Training Flow Debug ===")
                    trained_models = ModelTrainer.train_models(
                        train_data=st.session_state.train_data,
                        model_choice=st.session_state.model_choice,
                        model_size=st.session_state.model_size
                    )
                    print(f"Trained models: {list(trained_models.keys()) if trained_models else 'None'}")
                    
                    if trained_models:
                        st.session_state.trained_models = trained_models
                        backtests = perform_backtesting(
                            data=st.session_state.data,
                            test_data=st.session_state.test_data,
                            trained_models=trained_models,
                            horizon=st.session_state.forecast_horizon
                        )
                        
                        print("\nBacktest results:")
                        print(f"Backtests keys: {list(backtests.keys())}")
                        st.session_state.backtests = backtests
                        
                        st.success("Models trained and backtested successfully!")
                    else:
                        st.error("No models were successfully trained.")
                    
                except Exception as e:
                    st.error(f"Error in training flow: {str(e)}")
                    logger.error(f"Error in training flow: {str(e)}")
                    logger.error(traceback.format_exc())

            # Generate forecasts
            if st.session_state.forecast_button:
                try:
                    forecasts = generate_forecasts(
                        st.session_state.trained_models,
                        st.session_state.data,
                        st.session_state.forecast_horizon,
                        st.session_state.get('backtests', {})
                    )
                    
                    st.session_state.forecasts = forecasts
                    st.session_state.is_forecast_generated = True
                    st.success("Forecasts generated successfully!")
                except Exception as e:
                    st.error(f"An error occurred during forecast generation: {str(e)}")
                    logger.error(f"An error occurred during forecast generation: {str(e)}")
                    logger.error(traceback.format_exc())

            # Display results
            if st.session_state.is_forecast_generated:
                try:
                    # Debug logging
                    logger.debug("=== Debug Info ===")
                    logger.debug(f"Data type: {type(st.session_state.data)}")
                    logger.debug(f"Train data type: {type(st.session_state.train_data)}")
                    logger.debug(f"Test data type: {type(st.session_state.test_data)}")
                    logger.debug(f"Forecasts type: {type(st.session_state.forecasts)}")
                    logger.debug(f"Backtests type: {type(st.session_state.backtests)}")
                    logger.debug(f"Model metrics type: {type(st.session_state.model_metrics)}")
                        
                    display_results(
                        data=st.session_state.data,
                        train_data=st.session_state.train_data,
                        test_data=st.session_state.test_data,
                        forecasts=st.session_state.forecasts if isinstance(st.session_state.forecasts, dict) else {},
                        backtests=st.session_state.backtests if isinstance(st.session_state.backtests, dict) else {},
                        model_metrics=st.session_state.model_metrics if isinstance(st.session_state.model_metrics, dict) else {},
                        model_choice=st.session_state.model_choice
                    )
                except Exception as e:
                    st.error(f"An error occurred while displaying results: {str(e)}")
                    logger.error(f"An error occurred while displaying results: {str(e)}")
                    logger.error(traceback.format_exc())

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
