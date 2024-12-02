"""
Time Series Forecasting Streamlit Application
"""
import streamlit as st
import logging
from typing import Tuple
from darts import TimeSeries
import traceback
import plotly.graph_objects as go

from backend.data.data_loader import DataLoader
from backend.utils.app_components import ForecastingService
from backend.core.trainer import ModelTrainer
from backend.infrastructure.ui.components import UIComponents
from backend.utils.session_state import initialize_session_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeSeriesForecastApp:
    def __init__(self):
        """Initialize the application state."""
        self.session_state = st.session_state
        self.data_loader = DataLoader()

    def render(self):
        """Render the main application interface."""
        st.title("Time Series Forecasting App")

        # Data Loading Section
        st.header("1. Data Selection")
        data_option = st.radio(
            "Choose dataset:",
            ["Air Passengers", 
             "Monthly Milk Production", 
             "Electricity Consumption (Zurich)", 
             "Upload CSV"]
        )

        # Load and display data
        if data_option != self.session_state.get('last_data_option'):
            # Clear previous data if data source changed
            self.session_state.data = None
            self.session_state.last_data_option = data_option

        if 'data' not in self.session_state or self.session_state.data is None:
            data, train_data, test_data = self.data_loader.load_data()
            if data is not None:
                self.session_state.data = data
                self.session_state.train_data = train_data
                self.session_state.test_data = test_data
                
                # Display dataset information
                st.subheader("Dataset Information")
                st.write(f"Total data points: {len(data)}")
                st.write(f"Training data points: {len(train_data)}")
                st.write(f"Test data points: {len(test_data)}")
                
                # Plot the data
                st.subheader("Time Series Data")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data.time_index,
                    y=data.values().flatten(),
                    mode='lines',
                    name='Original Data'
                ))
                fig.add_trace(go.Scatter(
                    x=train_data.time_index,
                    y=train_data.values().flatten(),
                    mode='lines',
                    name='Training Data',
                    line=dict(dash='dash')
                ))
                fig.add_trace(go.Scatter(
                    x=test_data.time_index,
                    y=test_data.values().flatten(),
                    mode='lines',
                    name='Test Data',
                    line=dict(dash='dot')
                ))
                st.plotly_chart(fig)

        # Model Configuration (in sidebar)
        with st.sidebar:
            st.title("2. Model Configuration")
            model_choice = st.selectbox(
                "Select Model",
                ["All Models", "N-BEATS", "Prophet", "TiDE", "TSMixer"],
                key="model_choice"
            )
            
            model_size = st.selectbox(
                "Model Size",
                ["small", "medium", "large"],
                key="model_size"
            )

            if st.button("Train Model"):
                if self.train_models(model_choice, model_size):
                    st.success("Model trained successfully!")

            st.markdown("---")
            st.title("3. Forecasting")
            forecast_horizon = st.number_input(
                "Forecast Horizon",
                min_value=1,
                max_value=365,
                value=30
            )
            
            if st.button("Generate Forecast"):
                self.generate_forecast(forecast_horizon)

    def train_models(self, model_choice, model_size) -> bool:
        """Train selected models with the specified configuration."""
        try:
            # Add debug logging
            logger.info("Training models with choice: %s, size: %s", model_choice, model_size)
            logger.info("Train data type: %s", type(self.session_state.train_data))
            logger.info("Train data length: %s", 
                len(self.session_state.train_data) if self.session_state.train_data else 'None')
            
            trained_models = ModelTrainer.train_models(
                self.session_state.train_data,
                model_choice,
                model_size
            )
            if trained_models:
                self.session_state.trained_models = trained_models
                st.success("Models trained successfully!")
                return True
            return False
        except Exception as e:
            logger.error("Error training models: %s", str(e))
            logger.error(traceback.format_exc())
            st.error(f"An error occurred during model training: {str(e)}")
            return False

    def generate_forecast(self, forecast_horizon):
        try:
            logger.info("Generating forecast with horizon %s", forecast_horizon)
            logger.info("Test data type: %s", type(self.session_state.test_data))
            
            forecasts, backtests = ForecastingService.generate_forecasts(
                self.session_state.trained_models,
                self.session_state.data,
                forecast_horizon,
                self.session_state.test_data
            )
            self.session_state.forecasts = forecasts
            self.session_state.backtests = backtests
        except Exception as e:
            logger.error("Error in generate_forecast: %s", str(e))
            logger.error(traceback.format_exc())
            st.error(f"An error occurred during forecast generation: {str(e)}")

def main():
    initialize_session_state()
    app = TimeSeriesForecastApp()
    app.render()

if __name__ == "__main__":
    main()
