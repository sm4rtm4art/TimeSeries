import streamlit as st
import logging
from darts import TimeSeries

from backend.utils.session_state import initialize_session_state
from backend.utils.data_handling import load_data_if_needed
from backend.utils.ui_components import display_sidebar
from backend.models.training import train_models
from backend.models.forecasting import generate_forecasts
from backend.utils.app_components import display_results, display_results_without_test

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TimeSeriesApp:
    def __init__(self):
        initialize_session_state()

    def handle_training_and_forecasting(self):
        if st.session_state.train_button:
            train_models()
        if st.session_state.forecast_button and st.session_state.is_trained:
            generate_forecasts()

    def display_results_if_ready(self):
        if st.session_state.is_trained and st.session_state.is_forecast_generated:
            if isinstance(st.session_state.test_data, TimeSeries):
                display_results(
                    st.session_state.data,
                    st.session_state.forecasts,
                    st.session_state.test_data,
                    st.session_state.model_choice
                )
            else:
                st.warning("Test data is not available or not in the correct format. Displaying results without test data.")
                display_results_without_test(
                    st.session_state.data,
                    st.session_state.forecasts,
                    st.session_state.model_choice
                )
        elif not st.session_state.is_trained:
            st.info("Please train the models using the sidebar button.")
        elif not st.session_state.is_forecast_generated:
            st.info("Please generate forecasts using the sidebar button.")

    def run(self):
        logger.info("Starting the Time Series Forecasting application")
        st.set_page_config(page_title="Time Series Forecasting", layout="wide")
        st.title("Time Series Forecasting with Multiple Models")

        load_data_if_needed()
        display_sidebar()
        self.handle_training_and_forecasting()
        self.display_results_if_ready()

if __name__ == "__main__":
    app = TimeSeriesApp()
    app.run()