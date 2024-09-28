import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from darts import TimeSeries


def plot_forecast(historical_data, forecast, model_name, test_data=None):
    if not isinstance(historical_data, TimeSeries):
        st.error(f"Historical data is not a TimeSeries object. Type: {type(historical_data)}")
        return

    df = historical_data.pd_dataframe()
    df.columns = ['Historical']

    if test_data is not None and isinstance(test_data, TimeSeries):
        test_df = test_data.pd_dataframe()
        test_df.columns = ['Test']
        df = pd.concat([df, test_df], axis=1)

    if isinstance(forecast, tuple):
        forecast_df = forecast[0].pd_dataframe()  # Assuming the first element is the TimeSeries
    elif isinstance(forecast, TimeSeries):
        forecast_df = forecast.pd_dataframe()
    else:
        st.error(f"Forecast is not a TimeSeries object or tuple. Type: {type(forecast)}")
        return

    forecast_df.columns = [f'{model_name}_Forecast']

    # Combine all data
    combined_df = pd.concat([df, forecast_df], axis=1)

    fig = go.Figure()

    # Plot historical data
    fig.add_trace(go.Scatter(x=combined_df.index, y=combined_df['Historical'], name='Historical', line=dict(color='blue')))

    # Plot test data if available
    if 'Test' in combined_df.columns:
        fig.add_trace(go.Scatter(x=combined_df.index, y=combined_df['Test'], name='Test', line=dict(color='green')))

    # Plot forecast
    fig.add_trace(go.Scatter(x=combined_df.index, y=combined_df[f'{model_name}_Forecast'], name=f'{model_name} Forecast', line=dict(color='red')))

    fig.update_layout(title=f'{model_name} Forecast', xaxis_title='Date', yaxis_title='Value')
    st.plotly_chart(fig, use_container_width=True)


def plot_all_forecasts(historical_data, forecasts, test_data):
    fig = go.Figure()

    # Plot historical data
    fig.add_trace(go.Scatter(x=historical_data.time_index, y=historical_data.values().flatten(),
                             mode='lines', name='Historical'))

    # Plot test data
    fig.add_trace(go.Scatter(x=test_data.time_index, y=test_data.values().flatten(),
                             mode='lines', name='Test'))

    # Plot forecasts
    for model_name, forecast in forecasts.items():
        if model_name == "Chronos":
            # Handle Chronos forecast differently if needed
            # This is just an example, adjust based on actual Chronos output
            for i, f in enumerate(forecast):
                fig.add_trace(go.Scatter(x=f.time_index, y=f.values().flatten(),
                                         mode='lines', name=f'Chronos Forecast {i}'))
        else:
            fig.add_trace(go.Scatter(x=forecast.time_index, y=forecast.values().flatten(),
                                     mode='lines', name=f'{model_name} Forecast'))

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
