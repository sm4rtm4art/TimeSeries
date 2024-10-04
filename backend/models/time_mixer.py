from darts import TimeSeries
from darts.models import TSMixerModel
from darts.dataprocessing.transformers import Scaler
from typing import Union
import torch.nn as nn
import numpy as np

class TSMixerPredictor:
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        output_chunk_shift: int = 0,
        hidden_size: int = 64,
        ff_size: int = 64,
        num_blocks: int = 2,
        activation: str = "ReLU",
        dropout: float = 0.1,
        norm_type: Union[str, nn.Module] = "LayerNorm",
        normalize_before: bool = False,
        use_static_covariates: bool = True,
        **kwargs
    ) -> None:
        self.model = TSMixerModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
            hidden_size=hidden_size,
            ff_size=ff_size,
            num_blocks=num_blocks,
            activation=activation,
            dropout=dropout,
            norm_type=norm_type,
            normalize_before=normalize_before,
            use_static_covariates=use_static_covariates,
            force_reset=True,
            pl_trainer_kwargs={"accelerator": "auto", "precision": "32-true"},
            **kwargs
        )
        self.scaler = Scaler()
        self.data = None

    def train(self, data: TimeSeries) -> None:
        """
        Train the TSMixer model on the given data.

        :param data: Input data as a Darts TimeSeries
        """
        print(f"Training TSMixer model with data of length {len(data)}")
        scaled_data = self.scaler.fit_transform(data)
        scaled_data_32 = scaled_data.astype(np.float32)
        self.data = scaled_data_32
        self.model.fit(scaled_data_32)
        print("TSMixer model training completed")

    def predict(self, horizon: int, data: TimeSeries = None) -> TimeSeries:
        """
        Generate forecast using the trained model.

        :param horizon: Number of periods to forecast
        :param data: Historical data (optional, uses training data if not provided)
        :return: Forecast results as a TimeSeries object
        """
        if self.data is None or self.scaler is None:
            raise ValueError("Model has not been trained. Call train() before predict().")

        print(f"Predicting with TSMixer model. Horizon: {horizon}")
        
        if data is not None:
            scaled_data = self.scaler.transform(data)
            scaled_data_32 = scaled_data.astype(np.float32)
        else:
            scaled_data_32 = self.data
        
        forecast = self.model.predict(n=horizon, series=scaled_data_32)
        unscaled_forecast = self.scaler.inverse_transform(forecast)
        
        print(f"Generated forecast with length {len(unscaled_forecast)}")
        return unscaled_forecast
    
    def backtest(self, data: TimeSeries, forecast_horizon: int, start: int) -> TimeSeries:
        """
        Perform backtesting on the model.

        :param data: Input data as a Darts TimeSeries
        :param forecast_horizon: Number of periods to forecast in each iteration
        :param start: Start point for backtesting
        :return: Backtest results as a TimeSeries object
        """
        if self.data is None or self.scaler is None:
            raise ValueError("Model has not been trained. Call train() before backtest().")

        print(f"Backtesting TSMixer model. Data length: {len(data)}, Horizon: {forecast_horizon}, Start: {start}")

        scaled_data = self.scaler.transform(data)
        scaled_data_32 = scaled_data.astype(np.float32)
        backtest_results = self.model.historical_forecasts(
            series=scaled_data_32,
            start=start,
            forecast_horizon=forecast_horizon,
            stride=1,
            retrain=False,
            verbose=True,
            last_points_only=False
        )
        unscaled_backtest = self.scaler.inverse_transform(backtest_results)

        print(f"Backtest results length: {len(unscaled_backtest)}")
        return unscaled_backtest

def train_tsmixer_model(
    data: TimeSeries,
    input_chunk_length: int,
    output_chunk_length: int,
    **kwargs
) -> TSMixerPredictor:
    """
    Train a TSMixer model on the given data.

    :param data: Input data as a Darts TimeSeries
    :param input_chunk_length: The length of the input sequence
    :param output_chunk_length: The length of the output sequence
    :param kwargs: Additional keyword arguments for the TSMixerModel
    :return: Trained TSMixerPredictor instance
    """
    model = TSMixerPredictor(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        **kwargs
    )
    model.train(data)
    return model
