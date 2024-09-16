import streamlit as st
import pandas as pd

def plot_forecast(historical_data, forecast, model_name, test_data):
    df = historical_data.pd_dataframe()
    df.columns = ['Historical']
    
    # Add test data
    df = pd.concat([
        df,
        test_data.pd_dataframe().rename(columns={test_data.columns[0]: 'Test Data'})
    ])
    
    # Add forecast
    forecast_df = forecast.pd_dataframe().rename(columns={forecast.columns[0]: f'{model_name} Forecast'})
    df = pd.concat([df, forecast_df])
    
    # Ensure no duplicate indices
    df = df.groupby(df.index).first()
    
    st.line_chart(df)

def plot_all_forecasts(historical_data, forecasts, test_data):
    df = historical_data.pd_dataframe()
    df.columns = ['Historical']
    
    # Add test data
    df = pd.concat([
        df,
        test_data.pd_dataframe().rename(columns={test_data.columns[0]: 'Test Data'})
    ])
    
    # Add forecasts
    for model_name, forecast in forecasts.items():
        forecast_df = forecast.pd_dataframe().rename(columns={forecast.columns[0]: f'{model_name} Forecast'})
        df = pd.concat([df, forecast_df])
    
    # Ensure no duplicate indices
    df = df.groupby(df.index).first()
    
    st.line_chart(df)