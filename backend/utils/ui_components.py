import streamlit as st
from darts import TimeSeries
import logging

logger = logging.getLogger(__name__)


def display_sidebar():
    st.sidebar.title("Model Configuration")
    
    model_choice = st.sidebar.selectbox(
        "Choose a model",
        ("All Models", "N-BEATS", "Prophet", "TiDE", "Chronos", "TSMixer", "TFT")
    )
    
    model_size = st.sidebar.selectbox(
        "Choose model size",
        ("small", "medium", "large")
    )
    
    train_button = st.sidebar.button("Train Models")
    
    forecast_horizon = st.sidebar.number_input("Forecast Horizon", min_value=1, max_value=365, value=30)
    
    forecast_button = st.sidebar.button("Generate Forecast")
    
    # Log the values being returned for debugging
    logger.debug(f"display_sidebar returning: {model_choice}, {model_size}, {train_button}, {forecast_horizon}, {forecast_button}")
    
    return model_choice, model_size, train_button, forecast_horizon, forecast_button


def display_model_metrics():
    if 'model_metrics' in st.session_state:
        st.header("Model Metrics")
        for model_name, metrics in st.session_state.model_metrics.items():
            st.subheader(f"{model_name} Metrics")
            for metric_name, value in metrics.items():
                st.write(f"{metric_name}: {value:.4f}" if value is not None else f"{metric_name}: N/A")


def handle_visualize_split():
    if 'data' in st.session_state:
        # Display original data
        st.subheader("Original Data")
        st.line_chart(st.session_state.data.pd_dataframe())

        # Prepare and display train/test split
        train_data, test_data = prepare_data(st.session_state.data)
        from backend.utils.plotting import plot_train_test_data
        plot_train_test_data(train_data, test_data)
    else:
        st.warning("Please load data first.")

def handle_forecasting():
    if st.session_state.forecast_button:
        selected_model = st.session_state.trained_models[st.session_state.selected_model]
        forecast = generate_forecast_for_model(st.session_state.selected_model, selected_model, st.session_state.forecast_horizon)
        plot_forecast(st.session_state.data, forecast, st.session_state.selected_model, st.session_state.test_data)


def display_data_info(data: TimeSeries, train_data: TimeSeries, test_data: TimeSeries):
    st.subheader("Dataset Information")
    st.write(f"Start date: {data.start_time()}")
    st.write(f"End date: {data.end_time()}")
    st.write(f"Frequency: {data.freq}")
    st.write(f"Number of data points: {len(data)}")
    
    st.subheader("Original Data")
    st.line_chart(data.pd_dataframe())

    st.subheader("Train/Test Split")
    split_data = train_data.pd_dataframe()
    split_data["Test"] = test_data.pd_dataframe()
    st.line_chart(split_data)

