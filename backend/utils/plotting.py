import plotly.graph_objs as go
import plotly.express as px
from darts import TimeSeries
from typing import Dict, Union

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

    def plot_train_test_with_backtest(self, train_data: TimeSeries, test_data: TimeSeries, 
                                      forecasts: Dict[str, Dict[str, Union[TimeSeries, Dict]]], model_choice: str):
        fig = go.Figure()

        # Plot training data
        fig.add_trace(go.Scatter(x=train_data.time_index, y=train_data.values().flatten(),
                                 mode='lines', name='Training Data', line=dict(color='blue')))

        # Plot test data
        fig.add_trace(go.Scatter(x=test_data.time_index, y=test_data.values().flatten(),
                                 mode='lines', name='Test Data', line=dict(color='green')))

        # Plot backtests
        for i, (model_name, forecast_dict) in enumerate(forecasts.items()):
            if model_choice == "All Models" or model_name == model_choice:
                color = self.colors[i % len(self.colors)]
                if 'backtest' in forecast_dict and forecast_dict['backtest'] is not None:
                    backtest = forecast_dict['backtest']
                    if isinstance(backtest, TimeSeries):
                        x_values = backtest.time_index
                        y_values = backtest.values().flatten()
                        fig.add_trace(go.Scatter(x=x_values, y=y_values,
                                                 mode='lines', name=f'{model_name} Backtest', 
                                                 line=dict(color=color, dash='dash')))
                    else:
                        logger.warning(f"Unexpected backtest type for {model_name}: {type(backtest)}")

        fig.update_layout(title='Train/Test Split with Backtest',
                          xaxis_title='Date',
                          yaxis_title='Value',
                          legend_title='Legend',
                          hovermode='x unified')

        # Add a vertical line to separate train and test data
        last_train_date = train_data.time_index[-1]
        fig.add_vline(x=last_train_date, line_dash="dash", line_color="gray")

        return fig

    def plot_forecasts(self, data: TimeSeries, test_data: TimeSeries, 
                       forecasts: Dict[str, Dict[str, TimeSeries]], model_choice: str):
        fig = go.Figure()

        # Plot historical data
        fig.add_trace(go.Scatter(x=data.time_index, y=data.values().flatten(),
                                 mode='lines', name='Historical Data', line=dict(color='blue')))

        # Plot test data
        fig.add_trace(go.Scatter(x=test_data.time_index, y=test_data.values().flatten(),
                                 mode='lines', name='Test Data', line=dict(color='green')))

        # Plot forecasts
        for i, (model_name, forecast_dict) in enumerate(forecasts.items()):
            if model_choice == "All Models" or model_name == model_choice:
                color = self.colors[i % len(self.colors)]
                if 'future' in forecast_dict and forecast_dict['future'] is not None:
                    future_forecast = forecast_dict['future']
                    fig.add_trace(go.Scatter(x=future_forecast.time_index, y=future_forecast.values().flatten(),
                                             mode='lines', name=f'{model_name} Forecast', line=dict(color=color)))

        fig.update_layout(title='Forecasting Results',
                          xaxis_title='Date',
                          yaxis_title='Value',
                          legend_title='Legend',
                          hovermode='x unified')

        # Add a vertical line to separate historical and future data
        last_historical_date = data.time_index[-1]
        fig.add_vline(x=last_historical_date, line_dash="dash", line_color="gray")

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