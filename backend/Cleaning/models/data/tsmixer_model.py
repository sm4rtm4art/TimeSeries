import logging
import traceback

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import streamlit as st
import torch
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TCNModel  # We'll use TCN as base for TimeMixer
from pytorch_lightning.callbacks import EarlyStopping

logger = logging.getLogger(__name__)


def determine_accelerator():
    if torch.cuda.is_available():
        return "gpu"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class PrintEpochResults(pl.Callback):
    def __init__(self, progress_bar, status_text, total_epochs):
        super().__init__()
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.total_epochs = total_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        progress = (current_epoch + 1) / self.total_epochs
        self.progress_bar.progress(progress)
        self.status_text.text(f"Training Progress: {int(progress * 100)}%")


class TSMixerPredictor:
    def __init__(self, input_chunk_length: int = 24, output_chunk_length: int = 12):
        self.model_name = "TSMixer"
        self.n_epochs = 100
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.model = None
        self.scaler = Scaler()
        self.is_trained = False
        self.train_data = None

    def train(self, data: TimeSeries) -> None:
        """Train the TSMixer model."""
        try:
            st.text("Training TSMixer model...")
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Scale the data
            scaled_data = self.scaler.fit_transform(data.astype(np.float32))

            # Create callbacks
            early_stopping = EarlyStopping(
                monitor="train_loss",
                patience=5,
                min_delta=0.001,
                mode="min",
            )
            print_epoch = PrintEpochResults(progress_bar, status_text, self.n_epochs)

            # Initialize the model
            self.model = TCNModel(
                input_chunk_length=self.input_chunk_length,
                output_chunk_length=self.output_chunk_length,
                n_epochs=self.n_epochs,
                batch_size=32,
                n_layers=3,
                num_filters=64,
                kernel_size=3,
                dropout=0.1,
                random_state=42,
                pl_trainer_kwargs={
                    "accelerator": determine_accelerator(),
                    "callbacks": [early_stopping, print_epoch],
                    "enable_progress_bar": False,
                },
            )

            # Train the model
            self.model.fit(scaled_data)
            self.is_trained = True
            self.train_data = data
            st.success("TSMixer model trained successfully!")

        except Exception as e:
            logger.error(f"Error in TSMixer training: {str(e)}")
            logger.error(traceback.format_exc())
            st.error(f"Error training TSMixer model: {str(e)}")
            raise

    def predict(self, n: int, series: TimeSeries | None = None) -> TimeSeries:
        """Generate predictions for the given horizon.

        Args:
            n (int): Number of steps to forecast
            series (Optional[TimeSeries]): Optional series to use for prediction
                                         If None, uses the last training data

        Returns:
            TimeSeries: Forecasted values

        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        try:
            # Use provided series or last training data
            input_data = series if series is not None else self.train_data

            # Scale the input data
            scaled_data = self.scaler.transform(input_data.astype(np.float32))

            # Generate prediction
            prediction = self.model.predict(n=n, series=scaled_data)

            # Inverse transform the prediction
            return self.scaler.inverse_transform(prediction)

        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise

    def backtest(
        self,
        data: TimeSeries,
        start: float,
        forecast_horizon: int,
        stride: int = 1,
        retrain: bool = False,
    ) -> TimeSeries:
        """Perform backtesting."""
        if not self.is_trained:
            raise ValueError("Model must be trained before backtesting")
        try:
            # Scale the data
            scaled_data = self.scaler.transform(data.astype(np.float32))

            # Perform backtesting
            backtest_results = self.model.historical_forecasts(
                series=scaled_data,
                start=start,
                forecast_horizon=forecast_horizon,
                stride=stride,
                retrain=retrain,
            )

            # Inverse transform the results
            return self.scaler.inverse_transform(backtest_results)

        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}")
            logger.error(traceback.format_exc())
            raise


def display_results(
    data: TimeSeries,
    train_data: TimeSeries,
    test_data: TimeSeries,
    forecasts: dict[str, TimeSeries],
    backtests: dict[str, dict],
    model_metrics: dict[str, dict[str, float]],
    model_choice: str = "All Models",
) -> None:
    """Display the forecasting results in a Streamlit interface.

    Args:
        data: Full time series data
        train_data: Training portion of data
        test_data: Test portion of data
        forecasts: Dictionary of forecast results by model
        backtests: Dictionary of backtest results by model
        model_metrics: Performance metrics for each model
        model_choice: Selected model to display or "All Models"

    """
    try:
        st.subheader("Forecasting Results")

        # Display forecast plot
        st.write("### Forecast Visualization")

        # Create dataframe for display
        df_results = pd.DataFrame()
        df_results["Historical"] = data.pd_dataframe()

        # Add forecast data
        if model_choice == "All Models":
            for model_name, forecast in forecasts.items():
                df_results[f"{model_name} Forecast"] = forecast.pd_dataframe()
        else:
            if model_choice in forecasts:
                df_results[f"{model_choice} Forecast"] = forecasts[model_choice].pd_dataframe()

        # Show the plot
        st.line_chart(df_results)

        # Show metrics
        st.write("### Model Performance Metrics")
        metrics_df = pd.DataFrame(model_metrics).T.reset_index()
        metrics_df.columns = ["Model", "MAPE", "RMSE", "MSE"]
        st.dataframe(metrics_df.style.highlight_min(axis=0, subset=["MAPE", "RMSE", "MSE"]))

    except Exception as e:
        logger.error(f"Error displaying results: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Error displaying results: {str(e)}")
