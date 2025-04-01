"""Time Series Forecasting Streamlit Application."""

import logging
import traceback

import streamlit as st

from backend.data.data_loader import DataLoader
from backend.infrastructure.ui.components import UIComponents
from backend.utils.app_components import ForecastingService
from backend.utils.session_state import get_session_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesForecastApp:
    """Main application class for the Time Series Forecasting Streamlit app."""

    def __init__(self) -> None:
        """Initialize the application with session state variables."""
        # Initialize session state through the manager
        self.session_state = get_session_state()
        self.data_loader = DataLoader()
        # Initialize all session state variables
        if "last_data_option" not in st.session_state:
            st.session_state.last_data_option = None
        if "training_started" not in st.session_state:
            st.session_state.training_started = False
        if "model_choice" not in st.session_state:
            st.session_state.model_choice = None
        if "model_size" not in st.session_state:
            st.session_state.model_size = "small"

    def render(self) -> None:
        """Render the main application interface."""
        # Sidebar configuration
        with st.sidebar:
            st.header("Settings")
            test_size = st.slider(
                "Test Set Size (%)",
                min_value=10,
                max_value=40,
                value=20,
                help="Percentage of data to use for testing",
            )

            proceed = st.button("Continue with Selected Split", key="proceed_button")

            # Store the proceed state
            if proceed:
                st.session_state.training_started = True

            # Check session state instead of proceed
            if st.session_state.training_started:
                st.markdown("---")
                st.subheader("Model Configuration")

                # Use session state to maintain selected values
                model_choice = st.selectbox(
                    "Choose model(s):",
                    ["All Models", "N-BEATS", "Prophet", "TiDE", "TSMixer"],
                    key="sidebar_model_choice",
                    index=0
                    if st.session_state.model_choice is None
                    else ["All Models", "N-BEATS", "Prophet", "TiDE", "TSMixer"].index(st.session_state.model_choice),
                )

                model_size = st.selectbox(
                    "Model size:",
                    ["tiny", "small", "medium", "large"],
                    key="sidebar_model_size",
                    index=["tiny", "small", "medium", "large"].index(st.session_state.model_size),
                )

                # Store selections in session state
                st.session_state.model_choice = model_choice
                st.session_state.model_size = model_size

                if st.button("Train Models", key="train_button"):
                    self.train_models(model_choice, model_size)

        # Main content
        st.title("Time Series Forecasting App")

        # 1. Data Loading Section
        st.header("1. Data Selection")
        data_option = st.radio(
            "Choose dataset:",
            ["Air Passengers", "Monthly Milk Production", "Electricity Consumption (Zurich)", "Upload CSV"],
        )

        # Load data if needed
        if data_option != self.session_state.last_data_option or self.session_state.data is None:
            full_data, train_data, test_data = self.data_loader.load_data(data_option, test_size / 100)
            if full_data is not None:
                self.session_state.data = full_data
                self.session_state.train_data = train_data
                self.session_state.test_data = test_data
                self.session_state.last_data_option = data_option
                self.session_state.trained_models = None
                self.session_state.forecasts = None
                self.session_state.backtests = None

        # Display loaded data
        if self.session_state.data is not None:
            st.subheader("Dataset Preview")
            df = self.session_state.data.pd_dataframe()
            st.line_chart(df)

        # Remove duplicate model selection from main content
        # Only keep forecasting and results sections
        if self.session_state.trained_models:
            st.header("2. Forecasting")
            forecast_horizon = st.slider(
                "Forecast Horizon",
                min_value=1,
                max_value=100,
                value=30,
                key="forecast_horizon_slider",
            )

            if st.button("Generate Forecast", key="forecast_button"):
                self.generate_forecast(forecast_horizon)

        # Results Section
        if hasattr(self.session_state, "forecasts") and self.session_state.forecasts:
            self.display_results()

    def generate_forecast(self, forecast_horizon: int) -> None:
        """Generate forecasts for the loaded data using trained models.

        Args:
            forecast_horizon: The number of time steps to forecast

        """
        try:
            logger.info("Generating forecast with horizon %s", forecast_horizon)
            logger.info("Test data type: %s", type(self.session_state.test_data))

            # Add null checks before calling the function
            if (
                self.session_state.trained_models is None
                or self.session_state.data is None
                or self.session_state.test_data is None
            ):
                st.error("Cannot generate forecast: Missing data or trained models")
                return

            forecasts, backtests = ForecastingService.generate_forecasts(
                self.session_state.trained_models,
                self.session_state.data,
                forecast_horizon,
                self.session_state.test_data,
            )
            self.session_state.forecasts = forecasts
            self.session_state.backtests = backtests
        except Exception as e:
            logger.error("Error in generate_forecast: %s", str(e))
            logger.error(traceback.format_exc())
            st.error(f"An error occurred during forecast generation: {str(e)}")

    def display_results(self) -> None:
        """Display forecasting results using the UI components."""
        # Add null checks before calling the function
        if (
            self.session_state.data is None
            or self.session_state.train_data is None
            or self.session_state.test_data is None
            or self.session_state.forecasts is None
            or self.session_state.backtests is None
            or self.session_state.model_choice is None
        ):
            st.error("Cannot display results: Missing data")
            return

        UIComponents.display_results(
            data=self.session_state.data,
            train_data=self.session_state.train_data,
            test_data=self.session_state.test_data,
            forecasts=self.session_state.forecasts,
            backtests=self.session_state.backtests,
            model_choice=self.session_state.model_choice,
        )

    def train_models(self, model_choice: str, model_size: str) -> bool:
        """Train selected models with the specified configuration.

        Args:
            model_choice: The model or models to train
            model_size: The size/complexity of the model

        Returns:
            bool: True if training was successful, False otherwise

        """
        try:
            with st.spinner("Training models..."):
                logger.info(f"Training models with choice: {model_choice}, size: {model_size}")
                logger.info(f"Train data type: {type(self.session_state.train_data)}")

                # Add null check for train_data before calling len()
                if self.session_state.train_data is not None:
                    logger.info(f"Train data length: {len(self.session_state.train_data)}")
                else:
                    logger.warning("Train data is None, cannot calculate length")

                # Check for null data before attempting to train
                if self.session_state.train_data is None or self.session_state.test_data is None:
                    st.error("Cannot train models: Missing training or test data")
                    return False

                trained_models = ForecastingService.train_models(
                    train_data=self.session_state.train_data,
                    test_data=self.session_state.test_data,
                    model_choice=model_choice,
                    model_size=model_size,
                )

                if trained_models:
                    self.session_state.trained_models = trained_models
                    self.session_state.model_choice = model_choice
                    self.session_state.training_started = True
                    st.success("Models trained successfully!")
                    return True
                else:
                    st.error("No models were trained successfully.")
                    return False

        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            logger.error(traceback.format_exc())
            st.error(f"An error occurred during training: {str(e)}")
            return False


def main() -> None:
    """Initialize and run the Streamlit application."""
    app = TimeSeriesForecastApp()
    app.render()


if __name__ == "__main__":
    main()
