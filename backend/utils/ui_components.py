import streamlit as st


def display_sidebar():
    with st.sidebar:
        st.header("Model Settings")
        st.session_state.model_choice = st.selectbox("Choose Model", ["All Models", "N-BEATS", "Prophet", "TiDE", "Chronos"])
        if st.session_state.model_choice in ["All Models", "Chronos"]:
            st.session_state.model_size = st.selectbox("Chronos Model Size", ["tiny", "small", "medium", "large"])
        st.session_state.train_button = st.button("Train Models")

        # Always show forecast horizon and generate forecast button
        st.session_state.forecast_horizon = st.slider("Forecast Horizon (periods)", min_value=1, max_value=36, value=st.session_state.forecast_horizon)
        st.session_state.forecast_button = st.button("Generate Forecast")

        # Disable the forecast button if models are not trained
        if not st.session_state.is_trained:
            st.session_state.forecast_button = False

def display_model_metrics():
    if 'model_metrics' in st.session_state:
        st.header("Model Metrics")
        for model_name, metrics in st.session_state.model_metrics.items():
            st.subheader(f"{model_name} Metrics")
            st.write(metrics)

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
