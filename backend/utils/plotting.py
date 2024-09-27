import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def plot_forecast(historical_data, forecast, model_name, test_data):
    df = historical_data.pd_dataframe()
    df.columns = [f'Historical_{col}' for col in df.columns]

    test_df = test_data.pd_dataframe()
    test_df.columns = [f'Test_{col}' for col in test_df.columns]

    forecast_df = forecast.pd_dataframe()
    forecast_df.columns = [f'{model_name}_Forecast_{col}' for col in forecast_df.columns]

    df = pd.concat([df, test_df, forecast_df], axis=1)
    df = df.groupby(df.index).first()

    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col))

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
