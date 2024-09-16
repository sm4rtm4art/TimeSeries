import streamlit as st
import pandas as pd

from models.nbeats_model import train_nbeats_model, make_nbeats_forecast
from models.prophet_model import train_prophet_model, make_prophet_forecast
from models.tide_model import train_tide_model, make_tide_forecast
from utils.plotting import plot_forecast, plot_all_forecasts
from utils.metrics import calculate_metrics
from models.nbeats_model import NBEATSPredictor
from models.prophet_model import ProphetModel
from models.tide_model import train_tide_model

def train_models(train_data, model_choice):
    trained_models = {}
    models_to_train = ["N-BEATS", "Prophet", "TiDE"] if model_choice == "All Models" else [model_choice]
    
    for model in models_to_train:
        with st.spinner(f"Training {model} model... This may take a few minutes."):
            try:
                if model == "N-BEATS":
                    nbeats_model = NBEATSPredictor()
                    nbeats_model.train(train_data)
                    trained_models[model] = nbeats_model
                elif model == "Prophet":
                    prophet_model = ProphetModel()
                    prophet_model.train(train_data.pd_dataframe())
                    trained_models[model] = prophet_model
                else:  # TiDE
                    tide_model, scaler = train_tide_model(train_data)
                    trained_models[model] = (tide_model, scaler)
                st.success(f"{model} model trained successfully!")
            except Exception as e:
                st.error(f"Error training {model} model: {str(e)}")
    
    return trained_models


def generate_forecasts(trained_models, train_data, test_length, forecast_horizon):
    forecasts = {}
    for model, trained_model in trained_models.items():
        with st.spinner(f"Generating forecast for {model}... This may take a moment."):
            try:
                total_horizon = test_length + forecast_horizon
                if model == "N-BEATS":
                    forecasts[model] = make_nbeats_forecast(trained_model, total_horizon)
                elif model == "Prophet":
                    forecasts[model] = make_prophet_forecast(trained_model, total_horizon)
                else:  # TiDE
                    forecasts[model] = make_tide_forecast(*trained_model, total_horizon)
                
                # Ensure forecast starts after training data
                start_date = train_data.end_time() + train_data.freq
                forecasts[model] = forecasts[model].slice(start_date, forecasts[model].end_time())
                st.success(f"{model} forecast generated successfully!")
            except Exception as e:
                st.error(f"Error generating forecast for {model}: {str(e)}")
    return forecasts


def display_results(data, forecasts, test_data, model_choice):
    st.subheader("Train/Test Split")
    train_data = data.slice(data.start_time(), test_data.start_time() - data.freq)
    df_split = pd.concat([
        train_data.pd_dataframe().rename(columns={train_data.columns[0]: 'Training Data'}),
        test_data.pd_dataframe().rename(columns={test_data.columns[0]: 'Test Data'})
    ])
    st.line_chart(df_split)

    st.subheader("Forecast Results")
    if model_choice == "All Models":
        plot_all_forecasts(data, forecasts, test_data)
    else:
        if model_choice in forecasts:
            plot_forecast(data, forecasts[model_choice], model_choice, test_data)
        else:
            st.error(f"No forecast available for {model_choice}")
            return
    
    st.subheader("Forecast Metrics (Test Period)")
    metrics = {}
    for model, forecast in forecasts.items():
        test_forecast = forecast.slice(test_data.start_time(), test_data.end_time())
        metrics[model] = calculate_metrics(test_data, test_forecast)
    
    st.table(pd.DataFrame(metrics).T)