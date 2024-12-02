from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from darts import TimeSeries
import pandas as pd

from backend.api.schemas.requests import TimeSeriesData, ForecastRequest
from backend.domain.services.training import ModelTrainingService
from backend.domain.services.forecasting import ForecastingService
from backend.infrastructure.data.loader import DataLoader

router = APIRouter()

@router.post("/load_data")
async def load_data(time_series_data: TimeSeriesData) -> Dict[str, Any]:
    try:
        df = pd.DataFrame(time_series_data.data)
        df[time_series_data.date_column] = pd.to_datetime(df[time_series_data.date_column])
        df = df.set_index(time_series_data.date_column)
        time_series = TimeSeries.from_dataframe(df, value_cols=time_series_data.value_column)
        return {"message": "Data loaded successfully", "data_points": len(time_series)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/train_model")
async def train_model(forecast_request: ForecastRequest) -> Dict[str, str]:
    try:
        training_service = ModelTrainingService()
        result = await training_service.train(
            model_choice=forecast_request.model_choice,
            model_size=forecast_request.model_size
        )
        return {"message": "Model trained successfully", "model": forecast_request.model_choice}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/generate_forecast")
async def generate_forecast(forecast_request: ForecastRequest) -> Dict[str, Any]:
    try:
        forecasting_service = ForecastingService()
        result = await forecasting_service.generate_forecast(
            model_choice=forecast_request.model_choice,
            forecast_horizon=forecast_request.forecast_horizon
        )
        return {"forecasts": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) 