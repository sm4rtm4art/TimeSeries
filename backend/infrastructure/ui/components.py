import streamlit as st
import logging
from typing import Dict, Any
from backend.utils.plotting import TimeSeriesPlotter
from darts import TimeSeries

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
    def display_forecasts(data: TimeSeries, forecasts: Dict[str, Dict[str, TimeSeries]], model_choice: str):
        """Display forecasts for each model."""
        try:
            plotter = TimeSeriesPlotter()
            if model_choice == "All Models":
                for model_name, forecast_dict in forecasts.items():
                    plotter.plot_forecast(data, forecast_dict['future'], model_name)
        except Exception as e:
            logger.error(f"Error displaying forecasts: {str(e)}")
            st.error(f"Error displaying forecasts: {str(e)}")
