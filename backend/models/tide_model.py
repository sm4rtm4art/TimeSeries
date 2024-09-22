import numpy as np
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TiDEModel


def train_tide_model(data: TimeSeries) -> tuple[TiDEModel, Scaler]:
    """
    Train a TiDE model using the provided time series data.

    :param data: Input time series data
    :return: Tuple of (Trained TiDEModel instance, Scaler)
    """
    # Convert data to float32
    data_float32 = data.astype(np.float32)

    # Scale the data
    scaler = Scaler()
    scaled_data = scaler.fit_transform(data_float32)

    model = TiDEModel(
        input_chunk_length=24,
        output_chunk_length=12,
        decoder_output_dim=64,
        temporal_width_past=16,
        temporal_width_future=16,
        temporal_decoder_hidden=64,
        use_layer_norm=True,
        use_reversible_instance_norm=True,
        n_epochs=100,
        batch_size=32,
        optimizer_kwargs={'lr': 1e-3},
        random_state=42,
       # pl_trainer_kwargs={
       #     "callbacks": [
       #         EarlyStopping(
       #             monitor="train_loss",
       #             patience=5,
       #             min_delta=0.001,
       #             mode='min',
       #         )
       #     ]
       # }
    )
    model.fit(scaled_data)
    return model, scaler


def make_tide_forecast(model: TiDEModel,
                        scaler: Scaler,
                        forecast_horizon: int) -> TimeSeries:
    """
    Generate a forecast using the trained TiDE model.

    :param model: Trained TiDEModel instance
    :param scaler: Fitted Scaler instance
    :param forecast_horizon: Number of periods to forecast
    :return: Forecast as a TimeSeries object
    """
    forecast = model.predict(forecast_horizon)
    return scaler.inverse_transform(forecast)
