import logging
from darts import TimeSeries
from darts.models import Prophet
import pandas as pd
from darts.metrics import mape, rmse, mae
from tqdm import tqdm
import streamlit as st

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ProphetModel:
    def __init__(self, input_chunk_length=24, output_chunk_length=12):
        self.model = Prophet()
        self.data = None
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length

    def train(self, data: TimeSeries) -> None:
        logger.info(f"Training Prophet model with data of length {len(data)}")
        self.data = data
        try:
            self.model.fit(data)
            logger.info("Prophet model training completed")
        except Exception as e:
            logger.error(f"Error during Prophet model training: {str(e)}")
            raise

    def predict(self, horizon: int) -> TimeSeries:
        if self.data is None:
            raise ValueError("Model has not been trained. Call train() before predict().")

        logger.info(f"Predicting with Prophet model. Horizon: {horizon}")
        
        try:
            forecast = self.model.predict(horizon)
            logger.info(f"Generated forecast with length {len(forecast)}")
            return forecast
        except Exception as e:
            logger.error(f"Error during Prophet model prediction: {str(e)}")
            raise

    def historical_forecasts(self, series, start, forecast_horizon, stride=1, retrain=True, verbose=False):
        logger.info(f"Generating historical forecasts. Start: {start}, Horizon: {forecast_horizon}, Stride: {stride}")
        
        try:
            # Calculate the number of iterations
            n_iterations = (len(series) - start - forecast_horizon) // stride + 1
            
            # Create Streamlit progress bar and status text
            progress_bar = st.progress(0)
            status_text = st.empty()

            historical_forecasts = []
            for i in range(n_iterations):
                current_start = start + i * stride
                train_data = series[:current_start]
                if retrain:
                    self.train(train_data)
                forecast = self.predict(forecast_horizon)
                historical_forecasts.append(forecast)
                
                # Update progress bar and status text
                progress = (i + 1) / n_iterations
                progress_bar.progress(progress)
                status_text.text(f"Prophet Historical Forecasts: {i+1}/{n_iterations}")
            
            # Combine all forecasts into a single TimeSeries
            combined_forecast = TimeSeries.from_series(pd.concat([f.pd_series() for f in historical_forecasts]))
            logger.info(f"Generated historical forecasts with length {len(combined_forecast)}")
            return combined_forecast
        except Exception as e:
            logger.error(f"Error during historical forecasts generation: {str(e)}")
            raise

    def backtest(self, data: TimeSeries, forecast_horizon: int, start: int) -> TimeSeries:
        return self.historical_forecasts(data, start, forecast_horizon, retrain=True)

    def evaluate(self, actual: TimeSeries, predicted: TimeSeries) -> dict:
        try:
            return {
                'MAPE': mape(actual, predicted),
                'RMSE': rmse(actual, predicted),
                'MAE': mae(actual, predicted)
            }
        except Exception as e:
            logger.error(f"Error during Prophet model evaluation: {str(e)}")
            raise

def train_prophet_model(data: TimeSeries):
    model = ProphetModel()
    model.train(data)
    return model