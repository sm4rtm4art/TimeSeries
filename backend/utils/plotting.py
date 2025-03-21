import plotly.graph_objs as go
import plotly.express as px
from darts import TimeSeries
from typing import Dict, Union, Any
import logging
import traceback  # Add this import at the top
import streamlit as st
import pandas as pd
from backend.config.config_loader import config

logger = logging.getLogger(__name__)

class TimeSeriesPlotter:
    """Unified plotting class for time series visualization."""
    
    def __init__(self):
        """Initialize plotter with configuration."""
        self.config = config  # Load visualization config
    
    def plot_original_data(self, data: TimeSeries) -> None:
        """Plot original time series data."""
        try:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.time_index,
                y=data.values().flatten(),
                name='Original Data',
                line=self.config.line_styles['historical']
            ))
            
            layout = self.config.plot_layout.copy()
            layout.update({
                'title': 'Original Time Series Data',
                'xaxis_title': 'Date',
                'yaxis_title': 'Value'
            })
            fig.update_layout(**layout)
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error plotting original data: {str(e)}")
            logger.error(traceback.format_exc())
            st.error(f"Error plotting data: {str(e)}")

    def plot_unified_analysis(
        self,
        historical_data: TimeSeries,
        train_data: TimeSeries = None,
        test_data: TimeSeries = None,
        forecasts: Dict[str, Dict[str, TimeSeries]] = None,
        backtests: Dict[str, Dict[str, Union[TimeSeries, Dict[str, float]]]] = None,
        model_choice: str = "All Models"
    ) -> None:
        """Plot all time series data in separate views."""
        try:
            # 1. Plot Forecasts
            st.subheader("Future Forecasts")
            fig_forecast = go.Figure()

            # Plot historical/training data
            if train_data is not None:
                fig_forecast.add_trace(go.Scatter(
                    x=train_data.time_index,
                    y=train_data.values().flatten(),
                    name='Training Data',
                    line=self.config.line_styles['training']
                ))
            else:
                fig_forecast.add_trace(go.Scatter(
                    x=historical_data.time_index,
                    y=historical_data.values().flatten(),
                    name='Historical Data',
                    line=self.config.line_styles['historical']
                ))

            self._add_test_data(fig_forecast, test_data)
            self._add_forecasts(fig_forecast, forecasts, model_choice)

            # Update layout for forecast plot
            layout_forecast = self.config.plot_layout.copy()
            layout_forecast.update({
                'title': 'Future Forecasts',
                'xaxis_title': 'Date',
                'yaxis_title': 'Value'
            })
            fig_forecast.update_layout(**layout_forecast)
            st.plotly_chart(fig_forecast, use_container_width=True)

            # 2. Plot Backtests (if available)
            if backtests and any('backtest' in b for b in backtests.values()):
                st.subheader("Model Backtesting Results")
                fig_backtest = go.Figure()

                # Plot training data for backtest
                fig_backtest.add_trace(go.Scatter(
                    x=train_data.time_index,
                    y=train_data.values().flatten(),
                    name='Training Data',
                    line=self.config.line_styles['training']
                ))

                # Plot test data
                if test_data is not None:
                    fig_backtest.add_trace(go.Scatter(
                        x=test_data.time_index,
                        y=test_data.values().flatten(),
                        name='Actual Test Data',
                        line=self.config.line_styles['test']
                    ))

                # Add backtests
                self._add_backtests(fig_backtest, backtests, model_choice)

                # Update layout for backtest plot
                layout_backtest = self.config.plot_layout.copy()
                layout_backtest.update({
                    'title': 'Backtest Results (Model Performance on Test Data)',
                    'xaxis_title': 'Date',
                    'yaxis_title': 'Value'
                })
                fig_backtest.update_layout(**layout_backtest)
                st.plotly_chart(fig_backtest, use_container_width=True)

            # 3. Display Metrics Table
            if backtests:
                self._display_metrics(backtests)
                
            # 4. Display Error Analysis
            if backtests and test_data is not None:
                st.subheader("Error Analysis")
                for model_name, backtest_dict in backtests.items():
                    if 'metrics' in backtest_dict:
                        metrics = backtest_dict['metrics']
                        st.write(f"**{model_name}**")
                        cols = st.columns(3)
                        
                        # Handle MAPE metric
                        mape_value = metrics.get('MAPE', 'N/A')
                        mape_display = f"{mape_value:.2f}%" if isinstance(mape_value, (int, float)) else mape_value
                        cols[0].metric("MAPE", mape_display)
                        
                        # Handle RMSE metric
                        rmse_value = metrics.get('RMSE', 'N/A')
                        rmse_display = f"{rmse_value:.2f}" if isinstance(rmse_value, (int, float)) else rmse_value
                        cols[1].metric("RMSE", rmse_display)
                        
                        # Handle MAE metric
                        mae_value = metrics.get('MAE', 'N/A')
                        mae_display = f"{mae_value:.2f}" if isinstance(mae_value, (int, float)) else mae_value
                        cols[2].metric("MAE", mae_display)

        except Exception as e:
            logger.error(f"Error in unified plotting: {str(e)}")
            logger.error(traceback.format_exc())
            st.error(f"Error creating unified plot: {str(e)}")

    def plot_outliers(self, data: TimeSeries, outliers: TimeSeries) -> None:
        """Plot time series with detected outliers."""
        try:
            fig = go.Figure()
            
            # Plot original data
            fig.add_trace(go.Scatter(
                x=data.time_index,
                y=data.values().flatten(),
                name='Original Data',
                line=self.config.line_styles['historical']
            ))
            
            # Plot outliers
            outlier_indices = outliers.values().flatten().astype(bool)
            fig.add_trace(go.Scatter(
                x=data.time_index[outlier_indices],
                y=data.values().flatten()[outlier_indices],
                mode='markers',
                name='Outliers',
                marker=dict(color='red', size=8, symbol='x')
            ))
            
            layout = self.config.plot_layout.copy()
            layout.update({
                'title': 'Time Series with Detected Outliers',
                'xaxis_title': 'Date',
                'yaxis_title': 'Value'
            })
            fig.update_layout(**layout)
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error plotting outliers: {str(e)}")
            logger.error(traceback.format_exc())
            st.error(f"Error plotting outliers: {str(e)}")

    # Private helper methods
    def _add_test_data(self, fig: go.Figure, test_data: TimeSeries) -> None:
        """Add test data to the figure."""
        if test_data is not None:
            fig.add_trace(go.Scatter(
                x=test_data.time_index,
                y=test_data.values().flatten(),
                name='Test Data',
                line=self.config.line_styles['test']
            ))

    def _add_forecasts(self, fig: go.Figure, forecasts: Dict[str, Dict[str, TimeSeries]], model_choice: str) -> None:
        """Add forecasts to the figure."""
        if forecasts:
            for model_name, forecast_dict in forecasts.items():
                if model_choice != "All Models" and model_name != model_choice:
                    continue
                if 'future' in forecast_dict:
                    color = self.config.model_colors.get(model_name, '#000000')
                    line_style = dict(color=color, **self.config.line_styles['forecast'])
                    fig.add_trace(go.Scatter(
                        x=forecast_dict['future'].time_index,
                        y=forecast_dict['future'].values().flatten(),
                        name=f'{model_name} Forecast',
                        line=line_style
                    ))

    def _add_backtests(self, fig: go.Figure, backtests: Dict[str, Dict[str, Union[TimeSeries, Dict[str, float]]]], model_choice: str) -> None:
        """Add backtests to the figure."""
        if backtests:
            for model_name, backtest_dict in backtests.items():
                if model_choice != "All Models" and model_name != model_choice:
                    continue
                if isinstance(backtest_dict, dict) and 'backtest' in backtest_dict:
                    backtest = backtest_dict['backtest']
                    if isinstance(backtest, TimeSeries):
                        color = self.config.model_colors.get(model_name, '#000000')
                        line_style = dict(color=color, **self.config.line_styles['backtest'])
                        fig.add_trace(go.Scatter(
                            x=backtest.time_index,
                            y=backtest.values().flatten(),
                            name=f'{model_name} Backtest',
                            line=line_style
                        ))

    def _display_metrics(self, backtests: Dict[str, Dict[str, Union[TimeSeries, Dict[str, float]]]]) -> None:
        """Display metrics table if available."""
        try:
            metrics_data = {}
            for model_name, backtest_dict in backtests.items():
                if isinstance(backtest_dict, dict) and 'metrics' in backtest_dict:
                    metrics = backtest_dict['metrics']
                    if isinstance(metrics, dict):
                        metrics_data[model_name] = metrics
                        logger.debug(f"Added metrics for {model_name}: {metrics}")

            if metrics_data:
                st.subheader("Detailed Model Performance Metrics")
                metrics_df = pd.DataFrame(metrics_data).transpose()
                
                # Ensure all expected metrics exist in the DataFrame
                for metric in ['MAPE', 'RMSE', 'MAE']:
                    if metric not in metrics_df.columns:
                        metrics_df[metric] = 'N/A'
                
                # Apply styling with gradient colors - only to numeric columns
                numeric_columns = metrics_df.select_dtypes(include=['float64', 'int64']).columns
                subset = [col for col in ['MAPE', 'RMSE', 'MAE'] if col in numeric_columns]
                
                styled_df = metrics_df.style\
                    .format(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)\
                    .set_properties(**self.config.metrics_table_style)
                
                if subset:
                    styled_df = styled_df.background_gradient(cmap='RdYlGn_r', subset=subset)
                
                st.table(styled_df)
            else:
                logger.debug("No valid metrics found in backtest data")
                
        except Exception as e:
            logger.error(f"Error displaying metrics: {str(e)}")
            logger.error(traceback.format_exc())
            st.error(f"Error displaying metrics: {str(e)}")
