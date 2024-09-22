"""
Prophet model
"""

import pandas as pd
from darts import TimeSeries
from prophet import Prophet


class ProphetModel:
    def __init__(self):
        self.model = Prophet()

    def train(self, data: pd.DataFrame) -> None:
        """
        Train the Prophet model.

        :param data: Input data as a pandas DataFrame
        """
        df = self._prepare_data(data)
        self.model.fit(df)

    def forecast(self, periods: int) -> TimeSeries:
        """
        Generate forecast using the trained model.

        :param periods: Number of periods to forecast
        :return: Forecast results as a TimeSeries object
        """
        future = self._create_future_dataframe(periods)
        forecast = self.model.predict(future)
        return TimeSeries.from_dataframe(forecast[['ds', 'yhat']], time_col='ds', value_cols='yhat')

    @staticmethod
    def _prepare_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare the input data for Prophet model.

        :param data: Input data
        :return: Prepared data in the format required by Prophet
        """
        df = data.reset_index()
        df.columns = ['ds', 'y']
        return df

    def _create_future_dataframe(self, periods: int) -> pd.DataFrame:
        """
        Create a future dataframe for forecasting.

        :param periods: Number of periods to forecast
        :return: Future dataframe
        """
        return self.model.make_future_dataframe(periods=periods, freq='MS')


def train_prophet_model(data: TimeSeries) -> ProphetModel:
    model = ProphetModel()
    model.train(data.pd_dataframe())
    return model


def make_prophet_forecast(model: ProphetModel, horizon: int) -> TimeSeries:
    return model.forecast(horizon)
