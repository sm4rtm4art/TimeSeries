import streamlit as st
from backend.data.data_loader import DataLoader
from backend.data.data_preprocessing import detect_outliers
from backend.utils.app_components import (
    train_models, 
    generate_forecasts, 
    perform_backtesting, 
    display_results
)
from backend.utils.session_state import initialize_session_state
from backend.utils.ui_components import display_sidebar
from backend.utils.plotting import TimeSeriesPlotter
import logging
import traceback

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


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

        # Initialize forecasts
        forecasts = {}

        # Perform backtesting and generate forecasts
        if st.session_state.trained_models and forecast_button:
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
                    st.success("Forecasts generated and backtesting performed successfully!")
                except Exception as e:
                    st.error(f"An error occurred during forecast generation or backtesting: {str(e)}")
                    logger.error(f"An error occurred during forecast generation or backtesting: {str(e)}")
                    logger.error(traceback.format_exc())

        # Display results
        if st.session_state.is_forecast_generated:
            try:
                display_results(
                    st.session_state.data,
                    st.session_state.train_data,
                    st.session_state.test_data,
                    forecasts,
                    model_choice,
                    forecast_horizon,
                )
            except Exception as e:
                st.error(f"An error occurred while displaying results: {str(e)}")
                logger.error(f"An error occurred while displaying results: {str(e)}")
                logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
