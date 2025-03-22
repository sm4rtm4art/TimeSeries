"""Time Series Forecasting API

This module implements a FastAPI-based REST API for time series forecasting.
It provides endpoints for loading time series data, training forecasting models,
and generating forecasts using various models.

The API supports different forecasting models and allows for customization of
model parameters and forecast horizons. It uses the Darts library for time
series operations and forecasting.

Endpoints:
- GET /: Root endpoint returning a welcome message
- POST /load_data: Load time series data
- POST /train_model: Train a forecasting model
- POST /generate_forecast: Generate forecasts using a trained model

The module also includes Pydantic models for request validation and
type checking.

Note: This implementation uses in-memory storage for simplicity. For a
production environment, consider using a database for data persistence.
"""

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from darts import TimeSeries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.utils.app_components import generate_forecasts, train_models
from backend.utils.data_handling import prepare_data

# Add the project root directory to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

app = FastAPI()


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint that returns a welcome message.

    Returns:
        Dict[str, str]: A dictionary containing a welcome message.

    """
    return {"message": "Welcome to the Time Series Forecasting API"}


class TimeSeriesData(BaseModel):
    """Pydantic model for time series data input.

    Attributes:
        data (List[Dict[str, Any]]): List of dictionaries containing time series data.
        date_column (str): Name of the column containing date information.
        value_column (str): Name of the column containing value information.

    """

    data: list[dict[str, Any]]
    date_column: str
    value_column: str


class ForecastRequest(BaseModel):
    """Pydantic model for forecast request input.

    Attributes:
        model_choice (str): The chosen forecasting model.
        model_size (str): Size of the model (default is "small").
        forecast_horizon (int): Number of time steps to forecast.

    """

    model_choice: str
    model_size: str = "small"
    forecast_horizon: int


@app.post("/load_data")
async def load_data(time_series_data: TimeSeriesData) -> dict[str, Any]:
    """Endpoint to load time series data.

    Args:
        time_series_data (TimeSeriesData): Input time series data.

    Returns:
        Dict[str, Any]: A dictionary containing a success message and the number of data points.

    Raises:
        HTTPException: If there's an error in loading the data.

    """
    try:
        df = pd.DataFrame(time_series_data.data)
        df[time_series_data.date_column] = pd.to_datetime(df[time_series_data.date_column])
        df = df.set_index(time_series_data.date_column)
        time_series = TimeSeries.from_dataframe(df, value_cols=time_series_data.value_column)
        return {"message": "Data loaded successfully", "data_points": len(time_series)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/train_model")
async def train_model(forecast_request: ForecastRequest) -> dict[str, str]:
    """Endpoint to train a forecasting model.

    Args:
        forecast_request (ForecastRequest): Input forecast request data.

    Returns:
        Dict[str, str]: A dictionary containing a success message and the chosen model.

    Raises:
        HTTPException: If there's an error in training the model.

    """
    try:
        # Load data (you might want to store this in a database or cache)
        # For now, we'll use a dummy dataset
        data = TimeSeries.from_values(np.random.rand(100))

        train_data, test_data = prepare_data(data)
        trained_models = train_models(train_data, forecast_request.model_choice, forecast_request.model_size)
        return {"message": "Model trained successfully", "model": forecast_request.model_choice}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/generate_forecast")
async def generate_forecast(forecast_request: ForecastRequest) -> dict[str, Any]:
    """Endpoint to generate forecasts using a trained model.

    Args:
        forecast_request (ForecastRequest): Input forecast request data.

    Returns:
        Dict[str, Any]: A dictionary containing the generated forecasts.

    Raises:
        HTTPException: If there's an error in generating the forecast.

    """
    try:
        # Load data and trained model (you might want to store these in a database or cache)
        # For now, we'll use dummy data and retrain the model
        data = TimeSeries.from_values(np.random.rand(100))
        train_data, test_data = prepare_data(data)
        trained_models = train_models(train_data, forecast_request.model_choice, forecast_request.model_size)

        forecasts = generate_forecasts(trained_models, train_data, len(test_data), forecast_request.forecast_horizon)

        # Convert forecasts to a format that can be easily serialized
        serialized_forecasts = {
            model: forecast.pd_dataframe().to_dict(orient="records") for model, forecast in forecasts.items()
        }

        return {"forecasts": serialized_forecasts}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn

    # Use localhost for development (more secure)
    # For production deployment, consider using a reverse proxy like Nginx
    # or set environment-specific configuration
    host = "127.0.0.1"  # More secure than binding to all interfaces (0.0.0.0)
    if os.environ.get("ENVIRONMENT") == "production":
        # In production with proper security, we can bind to all interfaces
        # This allows Docker/Kubernetes networking to work
        host = "0.0.0.0"  # nosec B104 - Necessary for containerized environments

    uvicorn.run(app, host=host, port=8000)
