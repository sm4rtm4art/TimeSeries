import streamlit as st
from darts import TimeSeries
import logging
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Tuple, Any, Optional
import traceback
from backend.utils.plotting import TimeSeriesPlotter

logger = logging.getLogger(__name__)


def display_sidebar() -> Tuple[str, str, bool, int, bool]:
    """
    Display and handle sidebar components.
    
    Returns:
        Tuple containing:
        - model_choice (str): Selected model name
        - model_size (str): Model size (fixed to "small")
        - train_button (bool): Whether train button was clicked
        - forecast_horizon (int): Number of periods to forecast
        - forecast_button (bool): Whether forecast button was clicked
    """
    st.sidebar.title("Model Configuration")
    
    # Model selection
    model_choice = st.sidebar.selectbox(
        "Choose a model",
        ("All Models", "N-BEATS", "Prophet", "TiDE", "Chronos", "TSMixer", "TFT")
    )
    
    # Model size (fixed to "small" for now)
    model_size = "small"
    
    # Training button
    train_button = st.sidebar.button("Train Models")
    
    # Forecast horizon selection
    forecast_horizon = st.sidebar.number_input(
        "Forecast Horizon", 
        min_value=1, 
        max_value=365, 
        value=30
    )
    
    # Forecast button
    forecast_button = st.sidebar.button("Generate Forecast")
    
    # Debug logging
    logger.debug(f"Sidebar selections - Model: {model_choice}, Size: {model_size}, Train: {train_button}, Horizon: {forecast_horizon}, Forecast: {forecast_button}")
    
    return model_choice, model_size, train_button, forecast_horizon, forecast_button


def display_model_metrics(model_metrics: Dict[str, Dict[str, float]]) -> None:
    """Display metrics for each model."""
    if model_metrics:
        st.header("Model Metrics")
        for model_name, metrics in model_metrics.items():
            st.subheader(f"{model_name} Metrics")
            for metric_name, value in metrics.items():
                st.write(f"{metric_name}: {value:.4f}" if value is not None else f"{metric_name}: N/A")


def handle_visualize_split():
    if 'data' in st.session_state:
        # Display original data
        st.subheader("Original Data")
        st.line_chart(st.session_state.data.pd_dataframe())

        # Prepare and display train/test split
        train_data, test_data = prepare_data(st.session_state.data)
        from backend.utils.plotting import plot_train_test_data
        plot_train_test_data(train_data, test_data)
    else:
        st.warning("Please load data first.")

def handle_forecasting():
    if st.session_state.forecast_button:
        selected_model = st.session_state.trained_models[st.session_state.selected_model]
        forecast = generate_forecast_for_model(st.session_state.selected_model, selected_model, st.session_state.forecast_horizon)
        plot_forecast(st.session_state.data, forecast, st.session_state.selected_model, st.session_state.test_data)


def display_data_info(data: TimeSeries, train_data: TimeSeries, test_data: TimeSeries):
    st.subheader("Dataset Information")
    st.write(f"Start date: {data.start_time()}")
    st.write(f"End date: {data.end_time()}")
    st.write(f"Frequency: {data.freq}")
    st.write(f"Number of data points: {len(data)}")
    
    st.subheader("Original Data")
    st.line_chart(data.pd_dataframe())

    st.subheader("Train/Test Split")
    split_data = train_data.pd_dataframe()
    split_data["Test"] = test_data.pd_dataframe()
    st.line_chart(split_data)


def display_results(
    data: TimeSeries,
    train_data: TimeSeries,
    test_data: TimeSeries,
    forecasts: Dict[str, Dict[str, TimeSeries]],
    backtests: Dict[str, Dict],
    model_metrics: Dict[str, Dict[str, float]],
    model_choice: str = "All Models"
) -> None:
    """Display forecasting results including metrics."""
    try:
        tab1, tab2 = st.tabs(["Forecasts", "Backtesting"])
        
        # Tab 1: Forecasts
        with tab1:
            if forecasts:
                plotter = TimeSeriesPlotter()
                forecast_fig = plotter.plot_forecasts(
                    train_data=train_data,
                    test_data=test_data,
                    forecasts=forecasts,
                    model_choice=model_choice
                )
                st.plotly_chart(forecast_fig, use_container_width=True)
        
        # Tab 2: Backtesting
        with tab2:
            if backtests:
                # Plot backtests
                plotter = TimeSeriesPlotter()
                backtest_fig = plotter.plot_train_test_with_backtest(
                    train_data=train_data,
                    test_data=test_data,
                    backtests=backtests,
                    model_choice=model_choice
                )
                st.plotly_chart(backtest_fig, use_container_width=True)
                
                # Display metrics table
                st.subheader("Model Metrics")
                metrics_data = []
                
                for model_name, backtest_dict in backtests.items():
                    logger.info(f"Processing metrics for {model_name}")
                    if isinstance(backtest_dict, dict) and 'metrics' in backtest_dict:
                        metrics = backtest_dict['metrics']
                        metrics_data.append({
                            'Model': model_name,
                            'MAPE': f"{metrics.get('MAPE', 'N/A'):.2f}%",
                            'MSE': f"{metrics.get('MSE', 'N/A'):.2f}",
                            'RMSE': f"{metrics.get('RMSE', 'N/A'):.2f}"
                        })
                
                if metrics_data:
                    metrics_df = pd.DataFrame(metrics_data)
                    metrics_df.set_index('Model', inplace=True)
                    st.dataframe(metrics_df, use_container_width=True)
                else:
                    st.warning("No metrics available")
            else:
                st.info("No backtest results available")
                
    except Exception as e:
        logger.error(f"Error in display_results: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Error displaying results: {str(e)}")

def prepare_data(data: TimeSeries) -> Tuple[TimeSeries, TimeSeries]:
    """
    Prepare data by splitting into train and test sets.
    
    Args:
        data: Input TimeSeries data
        
    Returns:
        Tuple containing train and test TimeSeries
    """
    try:
        # Split at 80% of the data
        split_point = int(len(data) * 0.8)
        train_data = data[:split_point]
        test_data = data[split_point:]
        
        logger.debug(f"Data split - Train: {len(train_data)}, Test: {len(test_data)}")
        return train_data, test_data
        
    except Exception as e:
        logger.error(f"Error in prepare_data: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def generate_forecast_for_model(
    model_name: str,
    model: Any,
    forecast_horizon: int,
    data: TimeSeries
) -> TimeSeries:
    """
    Generate forecast for a specific model.
    
    Args:
        model_name: Name of the model
        model: Model instance
        forecast_horizon: Number of steps to forecast
        data: Input TimeSeries data
        
    Returns:
        Forecasted TimeSeries
    """
    try:
        logger.info(f"Generating forecast for {model_name}")
        forecast = model.predict(horizon=forecast_horizon, data=data)
        logger.info(f"Forecast generated successfully for {model_name}")
        return forecast
        
    except Exception as e:
        logger.error(f"Error generating forecast for {model_name}: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def plot_forecast(
    data: TimeSeries,
    forecast: TimeSeries,
    model_name: str,
    test_data: Optional[TimeSeries] = None
) -> None:
    """
    Plot the forecast results.
    
    Args:
        data: Original TimeSeries data
        forecast: Forecasted TimeSeries
        model_name: Name of the model
        test_data: Optional test TimeSeries data
    """
    try:
        # Plot the forecast
        st.subheader(f"{model_name} Forecast")
        st.line_chart(forecast.pd_dataframe())
        
        # Plot the test data
        if test_data is not None:
            st.subheader("Test Data")
            st.line_chart(test_data.pd_dataframe())
        
        # Plot the forecast vs test data
        st.subheader("Forecast vs Test Data")
        st.line_chart(forecast.pd_dataframe())
        st.line_chart(test_data.pd_dataframe())
        
        # Plot the forecast vs actual data
        st.subheader("Forecast vs Actual Data")
        st.line_chart(forecast.pd_dataframe())
        st.line_chart(data.pd_dataframe())
        
    except Exception as e:
        logger.error(f"Error plotting forecast for {model_name}: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"An error occurred while plotting forecast for {model_name}: {str(e)}")
