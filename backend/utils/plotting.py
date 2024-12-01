import plotly.graph_objs as go
import plotly.express as px
from darts import TimeSeries
from typing import Dict, Union, Any
import logging
import traceback  # Add this import at the top
import streamlit as st
import pandas as pd

logger = logging.getLogger(__name__)

class TimeSeriesPlotter:
    def __init__(self):
        self.colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]

    def plot_original_data(self, data: TimeSeries):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.time_index, y=data.values().flatten(),
                                 mode='lines', name='Original Data'))
        fig.update_layout(title='Original Time Series Data',
                          xaxis_title='Date',
                          yaxis_title='Value')
        return fig

    def plot_train_test_with_backtest(
        self,
        train_data: TimeSeries,
        test_data: TimeSeries,
        backtests: Dict[str, Dict[str, Union[TimeSeries, Dict[str, float]]]],
        model_choice: str = "All Models"
    ) -> go.Figure:
        try:
            # Debug logging
            logger.info("=== Plotting Backtests ===")
            logger.info(f"Models with backtests: {list(backtests.keys())}")
            
            fig = go.Figure()
            
            # Plot training data
            fig.add_trace(go.Scatter(
                x=train_data.time_index,
                y=train_data.values().flatten(),
                mode='lines',
                name='Training Data',
                line=dict(color='blue')
            ))
            
            # Plot test data
            fig.add_trace(go.Scatter(
                x=test_data.time_index,
                y=test_data.values().flatten(),
                mode='lines',
                name='Test Data',
                line=dict(color='green')
            ))
            
            # Plot backtests with debug info
            colors = ['red', 'purple', 'orange', 'brown', 'pink']
            for i, (model_name, backtest_dict) in enumerate(backtests.items()):
                logger.info(f"Processing backtest for {model_name}")
                
                if model_choice != "All Models" and model_name != model_choice:
                    continue
                    
                if isinstance(backtest_dict, dict) and 'backtest' in backtest_dict:
                    backtest = backtest_dict['backtest']
                    if isinstance(backtest, TimeSeries):
                        logger.info(f"Adding backtest plot for {model_name}")
                        logger.info(f"Backtest length: {len(backtest)}")
                        logger.info(f"Backtest time range: {backtest.start_time()} to {backtest.end_time()}")
                        
                        color = colors[i % len(colors)]
                        fig.add_trace(go.Scatter(
                            x=backtest.time_index,
                            y=backtest.values().flatten(),
                            mode='lines',
                            name=f'{model_name} Backtest',
                            line=dict(color=color, dash='dot')
                        ))
                    else:
                        logger.warning(f"Backtest for {model_name} is not a TimeSeries object")
                else:
                    logger.warning(f"Invalid backtest dictionary format for {model_name}")
            
            fig.update_layout(
                title='Time Series with Backtesting Results',
                xaxis_title='Date',
                yaxis_title='Value',
                showlegend=True,
                height=600,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting backtest: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def plot_forecasts(
        self,
        data: TimeSeries,
        test_data: TimeSeries,
        forecasts: Dict[str, Dict[str, TimeSeries]],
        model_choice: str = "All Models"
    ) -> go.Figure:
        """Plot forecast results."""
        try:
            logger.info(f"Starting plot_forecasts with models: {list(forecasts.keys())}")
            for model_name, forecast_dict in forecasts.items():
                logger.info(f"Forecast data for {model_name}: {forecast_dict.keys() if isinstance(forecast_dict, dict) else 'Not a dict'}")
            
            fig = go.Figure()

            # Plot historical data
            fig.add_trace(go.Scatter(
                x=data.time_index,
                y=data.values().flatten(),
                name='Historical Data',
                line=dict(color='blue')
            ))

            # Plot test data
            fig.add_trace(go.Scatter(
                x=test_data.time_index,
                y=test_data.values().flatten(),
                name='Test Data',
                line=dict(color='green')
            ))

            # Plot forecasts
            for i, (model_name, forecast_dict) in enumerate(forecasts.items()):
                if model_choice == "All Models" or model_name == model_choice:
                    if isinstance(forecast_dict, dict) and 'future' in forecast_dict:
                        future_forecast = forecast_dict['future']
                        if isinstance(future_forecast, TimeSeries):
                            color = self.colors[i % len(self.colors)]
                            fig.add_trace(go.Scatter(
                                x=future_forecast.time_index,
                                y=future_forecast.values().flatten(),
                                name=f'{model_name} Forecast',
                                line=dict(color=color, dash='dash')
                            ))

            # Update layout
            fig.update_layout(
                title='Forecasting Results',
                xaxis_title='Time',
                yaxis_title='Value',
                hovermode='x unified'
            )

            return fig

        except Exception as e:
            logger.error(f"Error in plot_forecasts: {str(e)}")
            logger.error(traceback.format_exc())
            # Create an empty figure in case of error
            fig = go.Figure()
            fig.add_annotation(text=f"Error plotting forecasts: {str(e)}", 
                              xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig

    def plot_outliers(self, data: TimeSeries, outliers: TimeSeries):
        fig = go.Figure()

        # Plot original data
        fig.add_trace(go.Scatter(x=data.time_index, y=data.values().flatten(),
                                 mode='lines', name='Original Data', line=dict(color='blue')))

        # Plot outliers
        outlier_indices = outliers.values().flatten().astype(bool)
        fig.add_trace(go.Scatter(x=data.time_index[outlier_indices],
                                 y=data.values().flatten()[outlier_indices],
                                 mode='markers', name='Outliers',
                                 marker=dict(color='red', size=8, symbol='x')))

        fig.update_layout(title='Time Series with Detected Outliers',
                          xaxis_title='Date',
                          yaxis_title='Value',
                          legend_title='Legend',
                          hovermode='x unified')

        return fig

    def plot_backtests(
        self,
        train_data: TimeSeries,
        test_data: TimeSeries,
        backtests: Dict[str, TimeSeries],
        model_choice: str = "All Models"
    ) -> go.Figure:
        """Plot backtest results."""
        try:
            fig = go.Figure()

            # Plot training data
            fig.add_trace(go.Scatter(
                x=train_data.time_index,
                y=train_data.values().flatten(),
                name='Training Data',
                line=dict(color='blue')
            ))

            # Plot test data
            fig.add_trace(go.Scatter(
                x=test_data.time_index,
                y=test_data.values().flatten(),
                name='Test Data',
                line=dict(color='green')
            ))

            # Plot backtest forecasts
            for i, (model_name, forecast) in enumerate(backtests.items()):
                if model_choice == "All Models" or model_name == model_choice:
                    color = self.colors[i % len(self.colors)]
                    fig.add_trace(go.Scatter(
                        x=forecast.time_index,
                        y=forecast.values().flatten(),
                        name=f'{model_name} Backtest',
                        line=dict(color=color, dash='dash')
                    ))

            # Update layout
            fig.update_layout(
                title='Model Backtesting Results',
                xaxis_title='Time',
                yaxis_title='Value',
                hovermode='x unified'
            )

            return fig

        except Exception as e:
            logger.error(f"Error in plot_backtests: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def plot_forecast(
        self,
        data: TimeSeries,
        forecast: TimeSeries,
        model_name: str
    ) -> None:
        """Plot the forecast against actual data."""
        try:
            fig = go.Figure()

            # Plot historical data
            fig.add_trace(go.Scatter(
                x=data.time_index,
                y=data.values().flatten(),
                mode='lines',
                name='Historical Data',
                line=dict(color='blue')
            ))

            # Plot forecast
            fig.add_trace(go.Scatter(
                x=forecast.time_index,
                y=forecast.values().flatten(),
                mode='lines',
                name=f'{model_name} Forecast',
                line=dict(color='red')
            ))

            # Update layout
            fig.update_layout(
                title=f'{model_name} Forecast',
                xaxis_title='Date',
                yaxis_title='Value',
                showlegend=True,
                hovermode='x unified'
            )

            # Display the plot in Streamlit
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            logger.error(f"Error plotting forecast: {str(e)}")
            logger.error(traceback.format_exc())
            st.error(f"Error plotting forecast: {str(e)}")

    def plot_backtest(
        self,
        data: TimeSeries,
        backtest: TimeSeries,
        model_name: str
    ) -> None:
        """Plot the backtest results against actual data."""
        try:
            fig = go.Figure()

            # Plot actual data
            fig.add_trace(go.Scatter(
                x=data.time_index,
                y=data.values().flatten(),
                mode='lines',
                name='Actual Data',
                line=dict(color='blue')
            ))

            # Plot backtest predictions
            fig.add_trace(go.Scatter(
                x=backtest.time_index,
                y=backtest.values().flatten(),
                mode='lines',
                name=f'{model_name} Backtest',
                line=dict(color='red')
            ))

            # Update layout
            fig.update_layout(
                title=f'{model_name} Backtest Results',
                xaxis_title='Date',
                yaxis_title='Value',
                showlegend=True,
                hovermode='x unified'
            )

            # Display the plot in Streamlit
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            logger.error(f"Error plotting backtest: {str(e)}")
            logger.error(traceback.format_exc())
            st.error(f"Error plotting backtest: {str(e)}")

    def plot_all_forecasts(
        self,
        data: TimeSeries,
        forecasts: Dict[str, Dict[str, TimeSeries]],
        model_choice: str = "All Models"
    ) -> None:
        """Plot all forecasts overlaid on the same graph."""
        try:
            fig = go.Figure()

            # Plot historical data
            fig.add_trace(go.Scatter(
                x=data.time_index,
                y=data.values().flatten(),
                mode='lines',
                name='Historical Data',
                line=dict(color='blue')
            ))

            # Plot forecasts for each model
            for i, (model_name, forecast_dict) in enumerate(forecasts.items()):
                if model_choice == "All Models" or model_name == model_choice:
                    if 'future' in forecast_dict:
                        color = self.colors[i % len(self.colors)]
                        fig.add_trace(go.Scatter(
                            x=forecast_dict['future'].time_index,
                            y=forecast_dict['future'].values().flatten(),
                            mode='lines',
                            name=f'{model_name} Forecast',
                            line=dict(color=color, dash='dash')
                        ))

            # Update layout
            fig.update_layout(
                title='Model Forecasts Comparison',
                xaxis_title='Date',
                yaxis_title='Value',
                showlegend=True,
                hovermode='x unified',
                height=600
            )

            # Display the plot
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            logger.error(f"Error plotting forecasts: {str(e)}")
            logger.error(traceback.format_exc())
            st.error(f"Error plotting forecasts: {str(e)}")

    def plot_all_backtests(
        self,
        data: TimeSeries,
        backtests: Dict[str, Dict[str, Union[TimeSeries, Dict[str, float]]]],
        model_choice: str = "All Models"
    ) -> None:
        """Plot all backtests overlaid on the same graph and display metrics."""
        try:
            fig = go.Figure()

            # Plot actual data
            fig.add_trace(go.Scatter(
                x=data.time_index,
                y=data.values().flatten(),
                mode='lines',
                name='Actual Data',
                line=dict(color='blue')
            ))

            # Plot backtests for each model
            for i, (model_name, backtest_dict) in enumerate(backtests.items()):
                if model_choice == "All Models" or model_name == model_choice:
                    if isinstance(backtest_dict, dict) and 'backtest' in backtest_dict:
                        color = self.colors[i % len(self.colors)]
                        fig.add_trace(go.Scatter(
                            x=backtest_dict['backtest'].time_index,
                            y=backtest_dict['backtest'].values().flatten(),
                            mode='lines',
                            name=f'{model_name} Backtest',
                            line=dict(color=color, dash='dash')
                        ))

            # Update layout
            fig.update_layout(
                title='Model Backtesting Comparison',
                xaxis_title='Date',
                yaxis_title='Value',
                showlegend=True,
                hovermode='x unified',
                height=600
            )

            # Display the plot
            st.plotly_chart(fig, use_container_width=True)

            # Display metrics table
            self.display_metrics_table(backtests)

        except Exception as e:
            logger.error(f"Error in plot_all_backtests: {str(e)}")
            logger.error(traceback.format_exc())
            st.error(f"Error plotting backtests: {str(e)}")

    def display_metrics_table(self, backtests: Dict[str, Dict[str, Any]]) -> None:
        """Display metrics in a formatted table."""
        try:
            metrics_data = []
            for model_name, backtest_dict in backtests.items():
                if isinstance(backtest_dict, dict) and 'metrics' in backtest_dict:
                    metrics = backtest_dict['metrics']
                    metrics_data.append({
                        'Model': model_name,
                        'MAPE (%)': f"{float(metrics.get('MAPE', 0)):.2f}",
                        'RMSE': f"{float(metrics.get('RMSE', 0)):.2f}",
                        'MSE': f"{float(metrics.get('MSE', 0)):.2f}"
                    })
            
            if metrics_data:
                st.subheader("Model Performance Metrics")
                df = pd.DataFrame(metrics_data)
                df.set_index('Model', inplace=True)
                
                # Style the dataframe
                def highlight_min(s):
                    is_min = pd.to_numeric(s.str.rstrip('%'), errors='coerce') == \
                            pd.to_numeric(s.str.rstrip('%'), errors='coerce').min()
                    return ['background-color: rgba(144, 238, 144, 0.3)' if v else '' for v in is_min]
                
                styled_df = df.style\
                    .apply(highlight_min)\
                    .set_properties(**{
                        'background-color': 'rgba(47, 47, 47, 0.8)',
                        'color': 'white',
                        'border': '1px solid gray'
                    })\
                    .format(precision=2)
                
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    height=150
                )
                
                # Add metrics explanation
                st.markdown("""
                **Metrics Explanation:**
                - **MAPE**: Mean Absolute Percentage Error (lower is better)
                - **RMSE**: Root Mean Square Error (lower is better)
                - **MSE**: Mean Square Error (lower is better)
                """)
            else:
                st.warning("No metrics available for display")

        except Exception as e:
            logger.error(f"Error displaying metrics table: {str(e)}")
            logger.error(traceback.format_exc())
            st.error(f"Error displaying metrics table: {str(e)}")

    def plot_forecasts_and_backtests(
        self,
        historical_data: TimeSeries,
        test_data: TimeSeries,
        forecasts: Dict[str, Dict[str, TimeSeries]],
        backtests: Dict[str, Dict[str, Union[TimeSeries, Dict[str, float]]]]
    ) -> None:
        """Plot all data, forecasts, and backtests in a single unified view."""
        
        fig = go.Figure()

        # Plot historical data
        fig.add_trace(go.Scatter(
            x=historical_data.time_index,
            y=historical_data.values().flatten(),
            name='Historical Data',
            line=dict(color='#1f77b4', width=2)
        ))

        # Plot test data
        if test_data is not None:
            fig.add_trace(go.Scatter(
                x=test_data.time_index,
                y=test_data.values().flatten(),
                name='Test Data',
                line=dict(color='#2ca02c', width=2)
            ))

        # Model-specific colors
        model_colors = {
            'N-BEATS': '#ff7f0e',
            'Prophet': '#d62728',
            'TiDE': '#9467bd',
            'TSMixer': '#8c564b'
        }

        # Plot backtests and forecasts for each model
        for model_name in forecasts.keys():
            color = model_colors.get(model_name, '#000000')
            
            # Plot backtest if available
            if model_name in backtests and 'backtest' in backtests[model_name]:
                backtest = backtests[model_name]['backtest']
                fig.add_trace(go.Scatter(
                    x=backtest.time_index,
                    y=backtest.values().flatten(),
                    name=f'{model_name} Backtest',
                    line=dict(color=color, dash='dot')
                ))
            
            # Plot forecast
            if 'future' in forecasts[model_name]:
                forecast = forecasts[model_name]['future']
                fig.add_trace(go.Scatter(
                    x=forecast.time_index,
                    y=forecast.values().flatten(),
                    name=f'{model_name} Forecast',
                    line=dict(color=color, dash='dash')
                ))

        # Update layout
        fig.update_layout(
            title='Time Series Forecasts and Backtests Comparison',
            xaxis_title='Date',
            yaxis_title='Value',
            template='plotly_dark',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0.5)"
            ),
            hovermode='x unified'
        )

        # Display plot
        st.plotly_chart(fig, use_container_width=True)

        # Display metrics table if available
        if backtests and any('metrics' in backtest_dict for backtest_dict in backtests.values()):
            st.subheader("Model Performance Metrics")
            metrics_df = pd.DataFrame({
                model: backtest_dict['metrics']
                for model, backtest_dict in backtests.items()
                if 'metrics' in backtest_dict
            }).transpose()
            
            # Format metrics to 4 decimal places
            st.table(metrics_df.style.format("{:.4f}"))

    def plot_unified_results(
        self,
        train_data: TimeSeries,
        test_data: TimeSeries,
        forecasts: Dict[str, Dict[str, TimeSeries]],
        backtests: Dict[str, Dict[str, Union[TimeSeries, Dict[str, float]]]],
        model_choice: str = "All Models"
    ) -> None:
        """Plot all forecasts and backtests in a single unified view."""
        try:
            fig = go.Figure()

            # Plot training data
            fig.add_trace(go.Scatter(
                x=train_data.time_index,
                y=train_data.values().flatten(),
                name='Training Data',
                line=dict(color='blue', width=2)
            ))

            # Plot test data
            fig.add_trace(go.Scatter(
                x=test_data.time_index,
                y=test_data.values().flatten(),
                name='Test Data',
                line=dict(color='green', width=2)
            ))

            # Define a consistent color scheme for models
            model_colors = {
                'N-BEATS': '#ff7f0e',  # Orange
                'Prophet': '#2ca02c',  # Green
                'TiDE': '#d62728',     # Red
                'TSMixer': '#9467bd',  # Purple
                'Chronos': '#8c564b'   # Brown
            }

            # Plot forecasts and backtests for each model
            for model_name in forecasts.keys():
                if model_choice == "All Models" or model_name == model_choice:
                    color = model_colors.get(model_name, '#000000')
                    
                    # Plot forecast if available
                    if 'future' in forecasts[model_name]:
                        forecast = forecasts[model_name]['future']
                        fig.add_trace(go.Scatter(
                            x=forecast.time_index,
                            y=forecast.values().flatten(),
                            name=f'{model_name} Forecast',
                            line=dict(color=color, dash='dash', width=1.5)
                        ))
                    
                    # Plot backtest if available
                    if model_name in backtests and 'backtest' in backtests[model_name]:
                        backtest = backtests[model_name]['backtest']
                        fig.add_trace(go.Scatter(
                            x=backtest.time_index,
                            y=backtest.values().flatten(),
                            name=f'{model_name} Backtest',
                            line=dict(color=color, dash='dot', width=1.5)
                        ))

            # Update layout
            fig.update_layout(
                title='Time Series Forecast with Backtesting',
                xaxis_title='Date',
                yaxis_title='Value',
                template='plotly_dark',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(0,0,0,0.5)"
                ),
                hovermode='x unified',
                height=600
            )

            # Display the plot
            st.plotly_chart(fig, use_container_width=True)

            # Display metrics table if available
            if any('metrics' in backtest_dict for backtest_dict in backtests.values()):
                st.subheader("Model Performance Metrics")
                metrics_data = {
                    model: backtest_dict['metrics']
                    for model, backtest_dict in backtests.items()
                    if 'metrics' in backtest_dict
                }
                metrics_df = pd.DataFrame(metrics_data).transpose()
                st.table(metrics_df.style.format("{:.4f}"))

        except Exception as e:
            logger.error(f"Error in unified plotting: {str(e)}")
            logger.error(traceback.format_exc())
            st.error(f"Error creating unified plot: {str(e)}")

    def plot_unified_forecasts(
        self,
        historical_data: TimeSeries,
        test_data: TimeSeries,
        forecasts: Dict[str, Dict[str, TimeSeries]],
        backtests: Dict[str, Dict[str, Union[TimeSeries, Dict[str, float]]]]
    ) -> None:
        """Plot all forecasts in a single unified view."""
        try:
            fig = go.Figure()

            # Plot historical data
            fig.add_trace(go.Scatter(
                x=historical_data.time_index,
                y=historical_data.values().flatten(),
                name='Historical Data',
                line=dict(color='black', width=2)
            ))

            # Define consistent colors for each model
            colors = {
                'N-BEATS': '#ff7f0e',
                'Prophet': '#2ca02c',
                'TiDE': '#d62728',
                'TSMixer': '#9467bd',
                'Chronos': '#8c564b'
            }

            # Plot forecasts (dashed lines)
            for model_name, forecast_dict in forecasts.items():
                if 'future' in forecast_dict:
                    forecast = forecast_dict['future']
                    color = colors.get(model_name, '#000000')
                    fig.add_trace(go.Scatter(
                        x=forecast.time_index,
                        y=forecast.values().flatten(),
                        name=f'{model_name} Forecast',
                        line=dict(color=color, dash='dash')
                    ))

            # Update layout
            fig.update_layout(
                title='Model Forecasts Comparison',
                xaxis_title='Date',
                yaxis_title='Value',
                template='plotly_dark',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(0,0,0,0.5)"
                ),
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            logger.error(f"Error in unified plotting: {str(e)}")
            logger.error(traceback.format_exc())
            st.error(f"Error creating unified plot: {str(e)}")

def display_forecasts(
    data: TimeSeries,
    forecasts: Dict[str, Dict[str, TimeSeries]],
    model_choice: str
) -> None:
    """Display forecasts for all models in a single plot."""
    try:
        plotter = TimeSeriesPlotter()
        plotter.plot_unified_forecasts(
            historical_data=data,
            test_data=None,
            forecasts=forecasts,
            backtests={}  # Empty dict since we're only showing forecasts
        )
    except Exception as e:
        logger.error(f"Error displaying forecasts: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Error displaying forecasts: {str(e)}")
