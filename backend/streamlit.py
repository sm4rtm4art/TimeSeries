import streamlit as st
import logging
from darts import TimeSeries
import traceback


from backend.utils.session_state import initialize_session_state
from backend.utils.data_handling import load_data_if_needed, prepare_data
from backend.utils.ui_components import display_sidebar
from backend.utils.app_components import train_models, generate_forecasts, display_results

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TimeSeriesApp:
    def __init__(self):
        initialize_session_state()

    def handle_training_and_forecasting(self):
        if st.session_state.train_button:
            st.session_state.train_data, st.session_state.test_data = prepare_data(st.session_state.data)
            st.session_state.trained_models = train_models(st.session_state.train_data, st.session_state.model_choice)
            st.session_state.is_trained = True

        if st.session_state.forecast_button and st.session_state.is_trained:
            st.session_state.forecasts = generate_forecasts(
                st.session_state.trained_models,
                st.session_state.data,
                st.session_state.forecast_horizon
            )
            st.session_state.is_forecast_generated = True

    def display_results_if_ready(self):
        if st.session_state.is_trained and st.session_state.forecasts:
            display_results(
                st.session_state.data,
                st.session_state.forecasts,
                st.session_state.test_data,
                st.session_state.model_choice,
                st.session_state.forecast_horizon  # Add this line
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