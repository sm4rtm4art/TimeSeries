import pandas as pd
import streamlit as st

from backend.models.chronos_model import train_chronos_model
from backend.models.forecasting import generate_forecast_for_model
from backend.models.nbeats_model import train_nbeats_model
from backend.models.prophet_model import train_prophet_model
from backend.models.tide_model import train_tide_model
from backend.utils.data_handling import prepare_data
from backend.utils.metrics import calculate_metrics

#logger = logging.getLogger(__name__)

def train_models():
    logger.info("Training models...")
    st.text("Training models...")
    try:
        st.session_state.train_data, st.session_state.test_data = prepare_data(st.session_state.data)
        model_choice = st.session_state.model_choice
        model_size = st.session_state.model_size

        if model_choice in ["All Models", "N-BEATS"]:
            train_and_store_model("N-BEATS", train_nbeats_model)
        if model_choice in ["All Models", "Prophet"]:
            train_and_store_model("Prophet", train_prophet_model)
        if model_choice in ["All Models", "TiDE"]:
            train_and_store_model("TiDE", train_tide_model)
        if model_choice in ["All Models", "Chronos"]:
            train_and_store_model("Chronos", train_chronos_model, model_size)

        st.session_state.is_trained = True
        # logger.info("Model(s) trained successfully!")
        st.success("Model(s) trained successfully!")

        # Calculate metrics for each trained model
        st.session_state.model_metrics = {}
        st.session_state.backtests = {}
        for model_name, model in st.session_state.trained_models.items():
            if hasattr(model, 'backtest'):
                forecast_horizon = st.session_state.forecast_horizon
                backtest_forecast, metrics = model.backtest(st.session_state.data, forecast_horizon)
                st.session_state.model_metrics[model_name] = metrics
                st.session_state.backtests[model_name] = backtest_forecast
            else:
                st.warning(f"Model {model_name} does not implement backtesting.")

        display_model_metrics()

    except Exception as e:
        # logger.error(f"Error during model training: {type(e).__name__}: {str(e)}")
        st.error(f"Error during model training: {type(e).__name__}: {str(e)}")
        st.exception(e)

def train_and_store_model(model_name, train_function, *args):
    # logger.info(f"Starting {model_name} model training")
    st.text(f"Starting {model_name} model training")
    model = train_function(st.session_state.train_data, *args)
    st.session_state.trained_models[model_name] = model
    # logger.info(f"{model_name} model training completed")
    st.text(f"{model_name} model training completed")

def display_model_metrics():
    st.subheader("Model Metrics")
    metrics_df = pd.DataFrame(st.session_state.model_metrics).T
    st.table(metrics_df)
