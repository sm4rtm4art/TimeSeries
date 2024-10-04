import streamlit as st


def display_sidebar():
    st.sidebar.title("Time Series Forecasting")
    
    model_choice = st.sidebar.selectbox(
        "Choose a model",
        ["All Models", "N-BEATS", "Prophet", "TiDE", "Chronos", "TSMixer", "TFT"]
    )
    
    model_size = st.sidebar.selectbox(
        "Choose model size (for Chronos)",
        ["tiny", "small", "medium", "large"]
    )
    
    forecast_horizon = st.sidebar.number_input("Forecast Horizon", min_value=1, value=30)
    
    train_button = st.sidebar.button("Train Models")
    
    return model_choice, model_size, forecast_horizon, train_button


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
