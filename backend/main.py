from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd

from darts import TimeSeries
from utils.data_handling import prepare_data
from utils.app_components import train_models, generate_forecasts
from models.chronos_model import ChronosPredictor
from models.nbeats_model import NBEATSPredictor
from models.prophet_model import ProphetModel
from models.tide_model import train_tide_model

app = FastAPI()

class TimeSeriesData(BaseModel):
    data: List[Dict[str, Any]]
    date_column: str
    value_column: str

class ForecastRequest(BaseModel):
    model_choice: str
    model_size: str = "small"
    forecast_horizon: int

@app.post("/load_data")
async def load_data(time_series_data: TimeSeriesData):
    try:
        df = pd.DataFrame(time_series_data.data)
        df[time_series_data.date_column] = pd.to_datetime(df[time_series_data.date_column])
        df = df.set_index(time_series_data.date_column)
        time_series = TimeSeries.from_dataframe(df, value_cols=time_series_data.value_column)
        return {"message": "Data loaded successfully", "data_points": len(time_series)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/train_model")
async def train_model(forecast_request: ForecastRequest):
    try:
        # Load data (you might want to store this in a database or cache)
        # For now, we'll use a dummy dataset
        data = TimeSeries.from_values(np.random.rand(100))
        
        train_data, test_data = prepare_data(data)
        trained_models = train_models(train_data, forecast_request.model_choice, forecast_request.model_size)
        return {"message": "Model trained successfully", "model": forecast_request.model_choice}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_forecast")
async def generate_forecast(forecast_request: ForecastRequest):
    try:
        # Load data and trained model (you might want to store these in a database or cache)
        # For now, we'll use dummy data and retrain the model
        data = TimeSeries.from_values(np.random.rand(100))
        train_data, test_data = prepare_data(data)
        trained_models = train_models(train_data, forecast_request.model_choice, forecast_request.model_size)
        
        forecasts = generate_forecasts(
            trained_models,
            train_data,
            len(test_data),
            forecast_request.forecast_horizon
        )
        
        # Convert forecasts to a format that can be easily serialized
        serialized_forecasts = {
            model: forecast.pd_dataframe().to_dict(orient="records")
            for model, forecast in forecasts.items()
        }
        
        return {"forecasts": serialized_forecasts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)