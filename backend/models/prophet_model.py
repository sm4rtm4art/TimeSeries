"""
Prophet model
This model is used to forecast time series data using the Prophet algorithm. The algorithm is provided by the Prophet library, 
a product of Facebook. Literature can be found here: https://facebook.github.io/prophet/docs/quick_start.html
"""

from darts import TimeSeries
from darts.models import Prophet
import pandas as pd

class ProphetModel:
    def __init__(self):
        self.model = Prophet()
        self.data = None

    def train(self, data: TimeSeries) -> None:
        """
        Train the Prophet model on the given data.

        :param data: Input data as a Darts TimeSeries
        """
        print(f"Training Prophet model with data of length {len(data)}")
        self.data = data
        self.model.fit(data)
        print("Prophet model training completed")

    def predict(self, horizon: int, data: TimeSeries = None) -> TimeSeries:
        """
        Generate forecast using the trained model.

        :param horizon: Number of periods to forecast
        :param data: Historical data (optional, uses training data if not provided)
        :return: Forecast results as a TimeSeries object
        """
        if self.data is None:
            raise ValueError("Model has not been trained. Call train() before predict().")

        print(f"Predicting with Prophet model. Horizon: {horizon}")
        
        # Ensure horizon is an integer
        if not isinstance(horizon, int):
            try:
                horizon = len(horizon)
            except TypeError:
                raise ValueError(f"Invalid horizon type: {type(horizon)}. Expected int or sequence.")

        # Get the last date from the training data
        last_date = self.data.end_time()

        # Create a future dataframe for Prophet
        future_dates = pd.date_range(start=last_date + self.data.freq, periods=horizon, freq=self.data.freq)
        future_df = pd.DataFrame({'ds': future_dates})

        # Make predictions
        forecast = self.model.predict(future_df)

        # Convert the forecast to a TimeSeries object
        forecast_ts = TimeSeries.from_dataframe(forecast, 'ds', 'yhat')
        
        print(f"Generated forecast with length {len(forecast_ts)}")
        return forecast_ts
    
    def backtest(self, data: TimeSeries, forecast_horizon: int, start: int) -> TimeSeries:
        """
        Perform backtesting on the model.

        :param data: Input data as a Darts TimeSeries
        :param forecast_horizon: Number of periods to forecast in each iteration
        :param start: Start point for backtesting
        :return: Backtest results as a TimeSeries object
        """
        if self.data is None:
            raise ValueError("Model has not been trained. Call train() before backtest().")

        print(f"Backtesting Prophet model. Data length: {len(data)}, Horizon: {forecast_horizon}, Start: {start}")

        # Remove duplicate timestamps
        data = data.drop_duplicates()

        # Check if frequency can be inferred
        if data.freq is None:
            print("Frequency not inferred, attempting to set frequency.")
            # Attempt to infer frequency
            try:
                data = data.with_frequency()
            except ValueError as e:
                print(f"Error inferring frequency: {str(e)}")
                # Handle the case where frequency cannot be inferred
                raise ValueError("Could not infer frequency. Please ensure data has a consistent frequency.")

        backtest_results = self.model.historical_forecasts(
            series=data,
            start=start,
            forecast_horizon=forecast_horizon,
            stride=1,
            retrain=True,
            verbose=True,
            last_points_only=False
        )

        print(f"Backtest results type: {type(backtest_results)}")
        if isinstance(backtest_results, list):
            print(f"Backtest results list length: {len(backtest_results)}")

            # Combine all backtest results into a single DataFrame
            combined_df = pd.concat([result.pd_series() for result in backtest_results])

            # Sort the index to ensure it's in chronological order
            combined_df = combined_df.sort_index()

            # Create TimeSeries with no specified frequency and fill missing dates
            try:
                backtest_ts = TimeSeries.from_series(
                    combined_df,
                    fill_missing_dates=True
                )
            except ValueError as e:
                print(f"Error creating TimeSeries: {str(e)}")
                # If that fails, try creating a TimeSeries without filling missing dates
                backtest_ts = TimeSeries.from_series(combined_df, fill_missing_dates=True, freq=None)
        else:
            backtest_ts = backtest_results

        print(f"Final backtest TimeSeries length: {len(backtest_ts)}")
        return backtest_ts

def train_prophet_model(data: TimeSeries):
    """
    Train a Prophet model on the given data.

    :param data: Input data as a Darts TimeSeries
    :return: Trained ProphetModel instance
    """
    model = ProphetModel()
    model.train(data)
    return model