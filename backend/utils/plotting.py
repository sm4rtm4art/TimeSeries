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
    df = historical_data.pd_dataframe()
    df.columns = [f'Historical_{col}' for col in df.columns]

    test_df = test_data.pd_dataframe()
    test_df.columns = [f'Test_{col}' for col in test_df.columns]

    for model_name, forecast in forecasts.items():
        forecast_df = forecast.pd_dataframe()
        forecast_df.columns = [f'{model_name}_Forecast_{col}' for col in forecast_df.columns]
        df = pd.concat([df, forecast_df], axis=1)

    df = pd.concat([df, test_df], axis=1)
    df = df.groupby(df.index).first()

    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col))

    fig.update_layout(title='All Forecasts', xaxis_title='Date', yaxis_title='Value')
    st.plotly_chart(fig, use_container_width=True)
