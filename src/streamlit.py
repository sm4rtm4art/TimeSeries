import streamlit as st
from data.data_loader import load_data
from utils.app_components import display_results, generate_forecasts, train_models
from utils.data_handling import prepare_data


def initialize_session_state():
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'train_data' not in st.session_state:
        st.session_state.train_data = None
    if 'test_data' not in st.session_state:
        st.session_state.test_data = None
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    if 'forecasts' not in st.session_state:
        st.session_state.forecasts = {}
    if 'is_trained' not in st.session_state:
        st.session_state.is_trained = False
    if 'is_forecast_generated' not in st.session_state:
        st.session_state.is_forecast_generated = False


def main():
    st.set_page_config(page_title="Time Series Forecasting", layout="wide")
    st.title("Time Series Forecasting with Multiple Models")

    initialize_session_state()

    # Load data
    st.session_state.data = load_data()

    # Sidebar
    st.sidebar.header("Model Settings")
    model_choice = st.sidebar.selectbox("Choose Model", ["All Models", "N-BEATS", "Prophet", "TiDE"])

    if st.session_state.data is not None:
        # Train models button
        if st.sidebar.button("Train Models"):
            st.session_state.train_data, st.session_state.test_data = prepare_data(st.session_state.data)
            st.session_state.trained_models = train_models(st.session_state.train_data, model_choice)
            st.session_state.is_trained = True
            st.success("Model(s) trained successfully!")

        # Generate forecast button
        if st.session_state.is_trained:
            forecast_horizon = st.sidebar.slider("Forecast Horizon (periods)", min_value=1, max_value=36, value=12)
            if st.sidebar.button("Generate Forecast"):
                st.session_state.forecasts = generate_forecasts(st.session_state.trained_models, st.session_state.train_data, len(st.session_state.test_data), forecast_horizon)
                st.session_state.is_forecast_generated = True
                st.success("Forecast(s) generated successfully!")

        # Main content area
        if not st.session_state.is_trained:
            st.info("Please train the models using the sidebar button.")
        elif not st.session_state.is_forecast_generated:
            st.info("Please generate forecasts using the sidebar button.")
        else:
            display_results(st.session_state.data, st.session_state.forecasts, st.session_state.test_data, model_choice)
    else:
        st.warning("No data loaded. Please select a dataset or upload a CSV file.")


if __name__ == "__main__":
    main()
