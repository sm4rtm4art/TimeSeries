"""App components"""

import logging
import traceback

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from darts import TimeSeries
from darts.metrics import mae, mape, rmse

# Add BasePredictor import
from backend.core.interfaces.base_model import TimeSeriesPredictor
from backend.core.trainer import ModelTrainer
from backend.utils.plotting import TimeSeriesPlotter

logger = logging.getLogger(__name__)


# Update the train_models function to use ModelTrainer
def train_models(train_data: TimeSeries, test_data: TimeSeries, model_choice: str, model_size: str = "small"):
    """Train selected models and perform backtesting."""
    try:
        return ModelTrainer.train_models(
            train_data=train_data,
            model_choice=model_choice,
            model_size=model_size,
        )
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        logger.error(traceback.format_exc())
        return None


def get_timedelta(series: TimeSeries, periods: int) -> pd.Timedelta:
    if len(series) < 2:
        raise ValueError("Series must have at least two data points to determine frequency.")

    # Try to infer frequency
    freq = pd.infer_freq(series.time_index)

    if freq is None:
        # If frequency can't be inferred, calculate the average time difference
        time_diff = series.time_index[-1] - series.time_index[0]
        avg_diff = time_diff / (len(series) - 1)
        return avg_diff * periods

    # Handle different frequency types
    if freq in ["D", "H", "T", "S"]:
        # For day, hour, minute, second frequencies
        return pd.Timedelta(periods, freq)
    elif freq in ["M", "MS"]:
        # For month frequencies
        return pd.offsets.MonthEnd(periods)
    elif freq in ["Y", "YS"]:
        # For year frequencies
        return pd.offsets.YearEnd(periods)
    elif freq == "W":
        # For week frequency
        return pd.Timedelta(weeks=periods)
    else:
        # For other frequencies, use the difference between first two timestamps
        time_diff = series.time_index[1] - series.time_index[0]
        return time_diff * periods


def calculate_backtest_start(data: TimeSeries, test_data: TimeSeries) -> float:
    """Calculate the starting point for backtesting."""
    # Use 60% of the data as the starting point instead of 80%
    return 0.6


def fallback_historical_forecasts(model, series, start, forecast_horizon, stride=1, retrain=False):
    historical_forecasts = []
    for i in range(start, len(series) - forecast_horizon + 1, stride):
        train_data = series[:i]
        if retrain:
            model.train(train_data)
        forecast = model.predict(horizon=forecast_horizon)
        historical_forecasts.append(forecast)

    # Combine all forecasts into a single TimeSeries
    combined_forecast = TimeSeries.from_series(pd.concat([f.pd_series() for f in historical_forecasts]))
    return combined_forecast


def generate_forecasts(
    trained_models,
    data: TimeSeries,
    forecast_horizon: int,
    backtests: dict[str, dict[str, TimeSeries | dict[str, float]]],
) -> dict[str, dict[str, TimeSeries]]:
    forecasts = {}

    if not isinstance(trained_models, dict):
        logger.error(f"Expected trained_models to be a dictionary, but got {type(trained_models)}")
        return forecasts

    for model_name, model in trained_models.items():
        print(f"Attempting to generate forecast for {model_name}")
        try:
            # Generate the forecast using horizon instead of n
            logger.info(f"Generating forecast for {model_name}")
            future_forecast = model.predict(horizon=forecast_horizon)

            # Generate future dates for the forecast
            future_dates = pd.date_range(
                start=data.end_time() + get_timedelta(data, 1),
                periods=forecast_horizon,
                freq=data.freq_str,
            )
            logger.info(f"Generated future dates for {model_name}: {future_dates}")

            # Ensure the forecast has the correct time index
            if len(future_forecast) == len(future_dates):
                future_forecast = TimeSeries.from_times_and_values(
                    future_dates,
                    future_forecast.values(),
                )
            else:
                logger.warning(
                    f"Forecast length ({len(future_forecast)}) doesn't match expected length ({len(future_dates)}). Using original forecast.",
                )

            forecasts[model_name] = {
                "future": future_forecast,
                "backtest": backtests[model_name]["backtest"] if model_name in backtests else None,
                "metrics": backtests[model_name]["metrics"] if model_name in backtests else None,
            }

            logger.info(f"Generated forecast for {model_name}: {future_forecast}")
            print(f"Successfully generated forecast for {model_name}")
        except Exception as e:
            logger.error(f"Error generating forecast for {model_name}: {str(e)}")
            logger.error(traceback.format_exc())
            print(f"Error generating forecast for {model_name}: {str(e)}")
    print(f"Generated forecasts for: {list(forecasts.keys())}")
    return forecasts


def display_results(
    data: TimeSeries,
    train_data: TimeSeries,
    test_data: TimeSeries,
    forecasts: dict[str, dict[str, TimeSeries]],
    backtests: dict[str, dict[str, TimeSeries | dict[str, float]]],
    model_metrics: dict[str, dict[str, float]],
    model_choice: str,
) -> None:
    """Display forecasting results and metrics."""
    try:
        plotter = TimeSeriesPlotter()

        # Create tabs for different visualizations
        tab1, tab2 = st.tabs(["Forecasts", "Backtests"])

        with tab1:
            st.subheader("Model Forecasts Comparison")
            plotter.plot_all_forecasts(data, forecasts, model_choice)

        with tab2:
            st.subheader("Model Backtests Comparison")
            if backtests:
                plotter.plot_all_backtests(data, backtests, model_choice)
            else:
                st.info("No backtest results available")

    except Exception as e:
        logger.error(f"Error in display_results: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Error displaying results: {str(e)}")


def perform_backtesting(
    data: TimeSeries,
    test_data: TimeSeries,
    trained_models: dict[str, TimeSeriesPredictor],
    horizon: int,
    stride: int = 1,
) -> dict[str, dict[str, TimeSeries | dict[str, float]]]:
    """Perform backtesting for all models."""
    backtests = {}

    try:
        # Calculate start point for backtesting (use float for relative position)
        start = 0.6  # Use last 40% of data for backtesting

        # Ensure we have enough data for backtesting
        total_length = len(data)
        min_required = int(total_length * start) + horizon
        if min_required > total_length:
            start = max(0.2, (total_length - horizon) / total_length)
            logger.warning(f"Adjusted backtesting start point to {start:.2f} to ensure enough data")

        for model_name, model in trained_models.items():
            try:
                logger.info(f"\nBacktesting {model_name}...")

                # Ensure scaler is fitted if model uses one
                if hasattr(model, "scaler") and hasattr(model.scaler, "fit"):
                    model.scaler.fit(data)

                backtest_result = model.backtest(
                    data=data,
                    start=start,
                    forecast_horizon=horizon,
                    stride=stride,
                )
                backtests[model_name] = backtest_result
                logger.info(f"Successfully completed backtesting for {model_name}")
                logger.info(f"Metrics for {model_name}: {backtest_result.get('metrics', {})}")

            except Exception as e:
                logger.error(f"Error in backtesting {model_name}: {str(e)}")
                logger.error(traceback.format_exc())
                continue

    except Exception as e:
        logger.error(f"Error in perform_backtesting: {str(e)}")
        logger.error(traceback.format_exc())

    return backtests


def calculate_metrics_for_all_models(
    actual: TimeSeries,
    forecasts: dict[str, dict[str, TimeSeries]],
) -> dict[str, dict[str, float | str]]:
    metrics = {}

    for model_name, forecast_dict in forecasts.items():
        logger.info(f"Calculating metrics for {model_name}")

        try:
            if "backtest" not in forecast_dict or forecast_dict["backtest"] is None:
                raise ValueError("No backtest available")

            backtest = forecast_dict["backtest"]

            # Ensure the time indices match
            if not actual.time_index.equals(backtest.time_index):
                logger.warning(f"Time indices don't match for {model_name}. Attempting to align...")
                common_indices = actual.time_index.intersection(backtest.time_index)
                if len(common_indices) == 0:
                    raise ValueError("No common time indices between actual and backtest data")

                actual_aligned = actual.slice(common_indices[0], common_indices[-1])
                backtest_aligned = backtest.slice(common_indices[0], common_indices[-1])
            else:
                actual_aligned = actual
                backtest_aligned = backtest

            logger.info(
                f"Computing metrics between series of lengths: actual={len(actual_aligned)}, backtest={len(backtest_aligned)}",
            )

            metrics[model_name] = {
                "MAPE": float(mape(actual_aligned, backtest_aligned)),
                "RMSE": float(rmse(actual_aligned, backtest_aligned)),
                "MAE": float(mae(actual_aligned, backtest_aligned)),
            }

            logger.info(f"Successfully calculated metrics for {model_name}: {metrics[model_name]}")

        except Exception as e:
            logger.error(f"Error calculating metrics for {model_name}: {str(e)}")
            metrics[model_name] = {
                "MAPE": str(e),
                "RMSE": str(e),
                "MAE": str(e),
            }

    return metrics


def debug_data_state(data: TimeSeries, test_data: TimeSeries, historical_forecasts: TimeSeries = None):
    """Helper function to debug data alignment issues"""
    print("=== Data State Debug ===")
    print(f"Full data range: {data.start_time()} to {data.end_time()}")
    print(f"Test data range: {test_data.start_time()} to {test_data.end_time()}")
    if historical_forecasts is not None:
        print(f"Historical forecasts range: {historical_forecasts.start_time()} to {historical_forecasts.end_time()}")
    print(f"Full data length: {len(data)}")
    print(f"Test data length: {len(test_data)}")
    if historical_forecasts is not None:
        print(f"Historical forecasts length: {len(historical_forecasts)}")
    print("=====================")


def display_metrics(model_metrics: dict[str, dict[str, float]]):
    """Display metrics for each model in a formatted way."""
    try:
        # Create a DataFrame to store all metrics
        metrics_data = []

        for model_name, metrics_dict in model_metrics.items():
            if isinstance(metrics_dict, dict):
                model_data = {"Model": model_name}
                for metric_name, value in metrics_dict.items():
                    if isinstance(value, (int, float)):
                        model_data[metric_name] = f"{value:.4f}"
                    else:
                        model_data[metric_name] = str(value)
                metrics_data.append(model_data)

        if metrics_data:
            # Convert to DataFrame and display as table
            metrics_df = pd.DataFrame(metrics_data)
            st.write("Model Performance Metrics:")
            st.table(metrics_df.set_index("Model"))
        else:
            st.warning("No metrics available")

    except Exception as e:
        logger.error(f"Error displaying metrics: {str(e)}")
        st.error("Error displaying metrics")


def display_forecasts(
    data: TimeSeries,
    forecasts: dict[str, dict[str, TimeSeries]],
    model_choice: str,
) -> None:
    """Display forecasts for all models in a single plot."""
    try:
        fig = go.Figure()

        # Plot historical data
        fig.add_trace(
            go.Scatter(
                x=data.time_index,
                y=data.values().flatten(),
                name="Historical Data",
                line=dict(color="black", width=2),
            ),
        )

        # Define consistent colors for each model
        colors = {
            "N-BEATS": "#ff7f0e",
            "Prophet": "#2ca02c",
            "TiDE": "#d62728",
            "TSMixer": "#9467bd",
            "Chronos": "#8c564b",
        }

        # Plot all forecasts in the same graph
        for model_name, forecast_dict in forecasts.items():
            if "future" in forecast_dict:
                forecast = forecast_dict["future"]
                color = colors.get(model_name, "#000000")
                fig.add_trace(
                    go.Scatter(
                        x=forecast.time_index,
                        y=forecast.values().flatten(),
                        name=f"{model_name} Forecast",
                        line=dict(color=color, dash="dash"),
                    ),
                )

        # Update layout
        fig.update_layout(
            title="Model Forecasts Comparison",
            xaxis_title="Date",
            yaxis_title="Value",
            template="plotly_dark",
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0.5)",
            ),
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        logger.error(f"Error displaying forecasts: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Error displaying forecasts: {str(e)}")


def plot_unified_analysis(
    self,
    historical_data: TimeSeries,
    train_data: TimeSeries = None,
    test_data: TimeSeries = None,
    forecasts: dict[str, dict[str, TimeSeries]] = None,
    backtests: dict[str, dict[str, TimeSeries | dict[str, float]]] = None,
    model_choice: str = "All Models",
) -> None:
    """Plot all time series data in a single unified view."""
    try:
        fig = go.Figure()

        # Plot historical/training data
        if train_data is not None:
            fig.add_trace(
                go.Scatter(
                    x=train_data.time_index,
                    y=train_data.values().flatten(),
                    name="Training Data",
                    line=config.line_styles["training"],
                ),
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=historical_data.time_index,
                    y=historical_data.values().flatten(),
                    name="Historical Data",
                    line=config.line_styles["historical"],
                ),
            )

        # Plot test data if available
        if test_data is not None:
            fig.add_trace(
                go.Scatter(
                    x=test_data.time_index,
                    y=test_data.values().flatten(),
                    name="Test Data",
                    line=config.line_styles["test"],
                ),
            )

        # Plot forecasts
        if forecasts:
            for model_name, forecast_dict in forecasts.items():
                if model_choice != "All Models" and model_name != model_choice:
                    continue

                if "future" in forecast_dict:
                    forecast = forecast_dict["future"]
                    color = config.model_colors.get(model_name, "#000000")
                    line_style = dict(color=color, **config.line_styles["forecast"])
                    fig.add_trace(
                        go.Scatter(
                            x=forecast.time_index,
                            y=forecast.values().flatten(),
                            name=f"{model_name} Forecast",
                            line=line_style,
                        ),
                    )

        # Plot backtests
        if backtests:
            for model_name, backtest_dict in backtests.items():
                if model_choice != "All Models" and model_name != model_choice:
                    continue

                if isinstance(backtest_dict, dict) and "backtest" in backtest_dict:
                    backtest = backtest_dict["backtest"]
                    if isinstance(backtest, TimeSeries):
                        color = config.model_colors.get(model_name, "#000000")
                        line_style = dict(color=color, **config.line_styles["backtest"])
                        fig.add_trace(
                            go.Scatter(
                                x=backtest.time_index,
                                y=backtest.values().flatten(),
                                name=f"{model_name} Backtest",
                                line=line_style,
                            ),
                        )

        # Update layout using config
        layout = config.plot_layout.copy()
        layout.update(
            {
                "title": "Time Series Analysis",
                "xaxis_title": "Date",
                "yaxis_title": "Value",
            },
        )
        fig.update_layout(**layout)

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        logger.error(f"Error in unified plotting: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Error creating unified plot: {str(e)}")


def generate_forecasts_and_backtests(
    trained_models: dict[str, TimeSeriesPredictor],
    data: TimeSeries,
    train_data: TimeSeries,
    test_data: TimeSeries,
    forecast_horizon: int,
) -> tuple[dict[str, dict[str, TimeSeries]], dict[str, dict[str, TimeSeries | dict[str, float]]]]:
    """Generate both forecasts and backtests for all models."""
    try:
        # Generate forecasts
        forecasts = ModelForecaster.generate_forecasts(
            models=trained_models,
            horizon=forecast_horizon,
            data=data,
        )

        # Generate backtests
        backtests = ModelBacktester.generate_backtests(
            models=trained_models,
            data=data,
            test_data=test_data,
            forecast_horizon=forecast_horizon,
        )

        return forecasts, backtests

    except Exception as e:
        logger.error(f"Error generating forecasts and backtests: {str(e)}")
        raise
