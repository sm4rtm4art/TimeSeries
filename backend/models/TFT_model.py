from darts import TimeSeries
from darts.models import TFTModel
from typing import Union, List, Tuple
import torch.nn as nn
from darts.dataprocessing.transformers import Scaler

class TFTPredictor:
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        output_chunk_shift: int = 0,
        hidden_size: int = 16,
        lstm_layers: int = 1,
        num_attention_heads: int = 4,
        full_attention: bool = False,
        feed_forward: str = 'GatedResidualNetwork',
        dropout: float = 0.1,
        hidden_continuous_size: int = 8,
        categorical_embedding_sizes: Union[List[Tuple[int, int]], None] = None,
        add_relative_index: bool = True,  
        loss_fn: Union[nn.Module, None] = None,
        likelihood: Union[str, None] = None,
        norm_type: str = 'LayerNorm',
        use_static_covariates: bool = True,
        **kwargs
    ) -> None:
        self.model = TFTModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
            hidden_size=hidden_size,
            lstm_layers=lstm_layers,
            num_attention_heads=num_attention_heads,
            full_attention=full_attention,
            feed_forward=feed_forward,
            dropout=dropout,
            hidden_continuous_size=hidden_continuous_size,
            categorical_embedding_sizes=categorical_embedding_sizes,
            add_relative_index=add_relative_index,
            loss_fn=loss_fn,
            likelihood=likelihood,
            norm_type=norm_type,
            use_static_covariates=use_static_covariates,
            force_reset=True,
            pl_trainer_kwargs={"accelerator": "auto", "precision": "32-true"},
            **kwargs
        )
        self.scaler = Scaler()
        self.data = None

    def train(self, data: TimeSeries) -> None:
        """
        Train the TFT model on the given data.

        :param data: Input data as a Darts TimeSeries
        """
        print(f"Training TFT model with data of length {len(data)}")
        self.data = data
        self.model.fit(data)
        print("TFT model training completed")

    def predict(self, horizon: int, data: TimeSeries = None) -> TimeSeries:
        """
        Generate forecast using the trained model.

        :param horizon: Number of periods to forecast
        :param data: Historical data (optional, uses training data if not provided)
        :return: Forecast results as a TimeSeries object
        """
        if self.data is None:
            raise ValueError("Model has not been trained. Call train() before predict().")

        print(f"Predicting with TFT model. Horizon: {horizon}")
        
        forecast = self.model.predict(n=horizon, series=data if data is not None else self.data)
        
        print(f"Generated forecast with length {len(forecast)}")
        return forecast
    
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

        print(f"Backtesting TFT model. Data length: {len(data)}, Horizon: {forecast_horizon}, Start: {start}")

        backtest_results = self.model.historical_forecasts(
            series=data,
            start=start,
            forecast_horizon=forecast_horizon,
            stride=1,
            retrain=False,
            verbose=True,
            last_points_only=False
        )

        print(f"Backtest results length: {len(backtest_results)}")
        return backtest_results

def train_tft_model(
    data: TimeSeries,
    input_chunk_length: int,
    output_chunk_length: int,
    **kwargs
) -> TFTPredictor:
    """
    Train a TFT model on the given data.

    :param data: Input data as a Darts TimeSeries
    :param input_chunk_length: The length of the input sequence
    :param output_chunk_length: The length of the output sequence
    :param kwargs: Additional keyword arguments for the TFTModel
    :return: Trained TFTPredictor instance
    """
    model = TFTPredictor(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        **kwargs
    )
    model.train(data)
    return model
