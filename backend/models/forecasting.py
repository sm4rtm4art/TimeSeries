import streamlit as st
import logging
from backend.utils.plotting import plot_forecast
from backend.models.chronos_model import make_chronos_forecast
from backend.models.nbeats_model import make_nbeats_forecast
from backend.models.prophet_model import make_prophet_forecast
from backend.models.tide_model import make_tide_forecast

logger = logging.getLogger(__name__)

def generate_forecasts():
    logger.info("Generating forecasts...")
    st.text("Generating forecasts...")
    forecasts_generated = False

    for model_name, model in st.session_state.trained_models.items():
        try:
            forecast = generate_forecast_for_model(model_name, model)
            if forecast is not None:
                st.session_state.forecasts[model_name] = forecast
                plot_forecast(st.session_state.train_data, st.session_state.test_data, forecast, model_name)
                forecasts_generated = True
        except Exception as e:
            logger.error(f"Error generating forecast for {model_name}: {type(e).__name__}: {str(e)}")
            st.error(f"Error generating forecast for {model_name}: {type(e).__name__}: {str(e)}")
            st.exception(e)

    if forecasts_generated:
        st.session_state.is_forecast_generated = True
        logger.info("Forecasts generated successfully!")
        st.success("Forecast(s) generated successfully!")
    else:
        logger.warning("No forecasts were generated")
        st.warning("No forecasts were generated")

def generate_forecast_for_model(model_name, model):
    forecast = None
    if model_name == "Prophet":
        forecast = make_prophet_forecast(model, st.session_state.forecast_horizon)
    elif model_name == "N-BEATS":
        forecast = make_nbeats_forecast(model, st.session_state.data, st.session_state.forecast_horizon)
    elif model_name == "TiDE":
        tide_model, tide_scaler = model
        forecast, error_msg = make_tide_forecast(tide_model, tide_scaler, st.session_state.data, st.session_state.forecast_horizon)
        if error_msg:
            logger.error(f"Failed to generate {model_name} forecast: {error_msg}")
            st.error(f"Failed to generate {model_name} forecast: {error_msg}")
            return None
    elif model_name == "Chronos":
        forecast = make_chronos_forecast(model, st.session_state.data, st.session_state.forecast_horizon)
    return forecast