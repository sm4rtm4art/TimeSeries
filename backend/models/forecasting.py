import logging
import traceback
from typing import Any, Dict

import streamlit as st
from darts import TimeSeries

from backend.models.chronos_model import make_chronos_forecast
from backend.models.nbeats_model import make_nbeats_forecast
from backend.models.prophet_model import make_prophet_forecast
from backend.models.tide_model import make_tide_forecast

logger = logging.getLogger(__name__)

def generate_forecast_for_model(model_name: str, model: Any, data: TimeSeries, forecast_horizon: int) -> TimeSeries:
    try:
        print(f"Generating forecast for {model_name}")
        if model_name == "TiDE":
            tide_model, scaler = model  # Unpack the tuple
            forecast = make_tide_forecast(tide_model, scaler, data, forecast_horizon)
        elif model_name == "N-BEATS":
            forecast = make_nbeats_forecast(model, data, forecast_horizon)
        elif model_name == "Prophet":
            forecast = make_prophet_forecast(model, forecast_horizon)
        elif model_name == "Chronos":
            forecast = make_chronos_forecast(model, data, forecast_horizon)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        if isinstance(forecast, TimeSeries):
            print(f"Generated forecast for {model_name}: Length = {len(forecast)}, Start time = {forecast.start_time()}, End time = {forecast.end_time()}")
            return forecast
        else:
            print(f"Forecast for {model_name} is not a TimeSeries object. Type: {type(forecast)}")
            raise ValueError(f"Forecast for {model_name} is not a TimeSeries object.")
    except Exception as e:
        error_msg = f"Error generating forecast for {model_name}: {type(e).__name__}: {str(e)}"
        print(error_msg)
        print("Traceback:")
        traceback.print_exc()
        raise

def generate_forecasts(trained_models: Dict[str, Any], data: TimeSeries, test_data: TimeSeries) -> Dict[str, TimeSeries]:
    forecasts = {}
    for model_name, model in trained_models.items():
        try:
            print(f"Generating forecast for {model_name}")
            if model_name == "TiDE":
                tide_model, scaler = model  # Unpack the tuple
                print(f"TiDE model: {tide_model}")
                print(f"Scaler: {scaler}")
                forecast = make_tide_forecast(tide_model, scaler, data, len(test_data))
            else:
                # ... (existing code for other models)
                pass

            if isinstance(forecast, TimeSeries):
                print(f"Generated forecast for {model_name}: Length = {len(forecast)}, Start time = {forecast.start_time()}, End time = {forecast.end_time()}")
                forecasts[model_name] = forecast
            else:
                print(f"Forecast for {model_name} is not a TimeSeries object. Type: {type(forecast)}")
                st.error(f"Forecast for {model_name} is not a TimeSeries object.")
        except Exception as e:
            error_msg = f"Error generating forecast for {model_name}: {type(e).__name__}: {str(e)}"
            print(error_msg)
            print("Traceback:")
            traceback.print_exc()
            st.error(error_msg)
    return forecasts
