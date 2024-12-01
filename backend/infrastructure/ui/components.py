import streamlit as st
import logging
from typing import Dict, Any, Union
from backend.utils.plotting import TimeSeriesPlotter
from darts import TimeSeries
import traceback

logger = logging.getLogger(__name__)

class UIComponents:
    @staticmethod
    def create_sidebar() -> Dict[str, Any]:
        """Display and handle sidebar components."""
        st.sidebar.title("Model Configuration")
        
        return {
            "model_choice": st.sidebar.selectbox(
                "Choose a model",
                ("All Models", "N-BEATS", "Prophet", "TiDE", "TSMixer")
            ),
            "model_size": "small",
            "train_button": st.sidebar.button("Train Models"),
            "forecast_horizon": st.sidebar.number_input(
                "Forecast Horizon", 
                min_value=1, 
                max_value=365, 
                value=30
            ),
            "forecast_button": st.sidebar.button("Generate Forecast")
        }

    @staticmethod
    def display_metrics(model_metrics: Dict[str, Dict[str, float]]):
        """Display metrics for each model."""
        try:
            metrics_data = []
            for model_name, metrics_dict in model_metrics.items():
                if isinstance(metrics_dict, dict):
                    model_data = {'Model': model_name}
                    for metric_name, value in metrics_dict.items():
                        model_data[metric_name] = f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
                    metrics_data.append(model_data)
            
            if metrics_data:
                st.write("Model Performance Metrics:")
                st.table(pd.DataFrame(metrics_data).set_index('Model'))
            else:
                st.warning("No metrics available")
        except Exception as e:
            logger.error(f"Error displaying metrics: {str(e)}")
            st.error("Error displaying metrics")

    @staticmethod
    def display_forecasts(data: TimeSeries, forecasts: Dict[str, Dict[str, TimeSeries]], model_choice: str) -> None:
        """Display forecasts for all models in a single plot."""
        try:
            plotter = TimeSeriesPlotter()
            plotter.plot_unified_analysis(
                historical_data=data,
                forecasts=forecasts,
                model_choice=model_choice
            )
        except Exception as e:
            logger.error(f"Error displaying forecasts: {str(e)}")
            logger.error(traceback.format_exc())
            st.error(f"Error displaying forecasts: {str(e)}")

    @staticmethod
    def display_results(
        data: TimeSeries,
        train_data: TimeSeries,
        test_data: TimeSeries,
        forecasts: Dict[str, Dict[str, TimeSeries]],
        backtests: Dict[str, Dict[str, Union[TimeSeries, Dict[str, float]]]],
        model_choice: str = "All Models"
    ) -> None:
        """Display all results including forecasts, backtests, and metrics.
        
        Note: model_metrics are now included in the backtests dictionary
        and don't need to be passed separately.
        """
        try:
            plotter = TimeSeriesPlotter()
            plotter.plot_unified_analysis(
                historical_data=data,
                train_data=train_data,
                test_data=test_data,
                forecasts=forecasts,
                backtests=backtests,
                model_choice=model_choice
            )
        except Exception as e:
            logger.error(f"Error in display_results: {str(e)}")
            st.error(f"Error displaying results: {str(e)}")
