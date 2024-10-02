import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from darts import TimeSeries
import traceback



def plot_forecast(historical_data: TimeSeries, forecast: TimeSeries, model_name: str, test_data: TimeSeries):
    df = historical_data.pd_dataframe()
    df.columns = [f'Historical_{col}' for col in df.columns]

    test_df = test_data.pd_dataframe()
    test_df.columns = [f'Test_{col}' for col in test_df.columns]

    forecast_df = forecast.pd_dataframe()
    forecast_df.columns = [f'{model_name}_Forecast_{col}' for col in forecast_df.columns]

    # Combine all dataframes
    combined_df = pd.concat([df, test_df, forecast_df], axis=1)
    combined_df = combined_df.groupby(combined_df.index).first()

    fig = go.Figure()
    for col in combined_df.columns:
        fig.add_trace(go.Scatter(x=combined_df.index, y=combined_df[col], name=col))

    fig.update_layout(title=f'{model_name} Forecast', xaxis_title='Date', yaxis_title='Value')
    st.plotly_chart(fig, use_container_width=True)


def plot_all_forecasts(historical_data: TimeSeries, forecasts: dict, test_data: TimeSeries):
    df = historical_data.pd_dataframe()
    df.columns = [f'Historical_{col}' for col in df.columns]

    test_df = test_data.pd_dataframe()
    test_df.columns = [f'Test_{col}' for col in test_df.columns]

    # Combine all forecast dataframes
    forecast_dfs = []
    for model_name, forecast in forecasts.items():
        forecast_df = forecast.pd_dataframe()
        forecast_df.columns = [f'{model_name}_Forecast_{col}' for col in forecast_df.columns]
        forecast_dfs.append(forecast_df)

    # Combine all dataframes
    combined_df = pd.concat([df] + forecast_dfs + [test_df], axis=1)
    combined_df = combined_df.groupby(combined_df.index).first()

    fig = go.Figure()
    for col in combined_df.columns:
        fig.add_trace(go.Scatter(x=combined_df.index, y=combined_df[col], name=col))

    fig.update_layout(title='All Forecasts', xaxis_title='Date', yaxis_title='Value')
    st.plotly_chart(fig, use_container_width=True)


def plot_all_forecasts_without_test(historical_data, forecasts):
    fig = go.Figure()

    # Plot historical data
    fig.add_trace(go.Scatter(x=historical_data.time_index, y=historical_data.values().flatten(),
                             mode='lines', name='Historical'))

    # Plot forecasts
    for model_name, forecast in forecasts.items():
        if model_name == "Chronos":
            # Handle Chronos forecast differently if needed
            for i, f in enumerate(forecast):
                fig.add_trace(go.Scatter(x=f.time_index, y=f.values().flatten(),
                                         mode='lines', name=f'Chronos Forecast {i}'))
        else:
            fig.add_trace(go.Scatter(x=forecast.time_index, y=forecast.values().flatten(),
                                     mode='lines', name=f'{model_name} Forecast'))

    fig.update_layout(title='All Forecasts', xaxis_title='Date', yaxis_title='Value')
    st.plotly_chart(fig, use_container_width=True)


def plot_train_test_data(train_data: TimeSeries, test_data: TimeSeries):
    st.subheader("Train/Test Split")
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=train_data.time_index, y=train_data.values().flatten(),
                             mode='lines', name='Training Data'))
    fig.add_trace(go.Scatter(x=test_data.time_index, y=test_data.values().flatten(),
                             mode='lines', name='Test Data'))
    
    fig.update_layout(title='Train/Test Data', xaxis_title='Date', yaxis_title='Value')
    st.plotly_chart(fig, use_container_width=True)


def plot_train_test_forecasts(historical_data: TimeSeries, test_data: TimeSeries, forecasts: Dict[str, TimeSeries], model_choice: str):
    fig = go.Figure()

    # Plot historical data
    fig.add_trace(go.Scatter(x=historical_data.time_index, y=historical_data.values().flatten(),
                             mode='lines', name='Historical'))

    # Plot test data
    fig.add_trace(go.Scatter(x=test_data.time_index, y=test_data.values().flatten(),
                             mode='lines', name='Test Data'))

    # Plot forecasts
    for model_name, forecast in forecasts.items():
        if model_choice == "All Models" or model_name == model_choice:
            fig.add_trace(go.Scatter(x=forecast.time_index, y=forecast.values().flatten(),
                                     mode='lines', name=f'{model_name} Forecast'))

    fig.update_layout(title='Train/Test Split and Forecasts', xaxis_title='Date', yaxis_title='Value')
    st.plotly_chart(fig, use_container_width=True)