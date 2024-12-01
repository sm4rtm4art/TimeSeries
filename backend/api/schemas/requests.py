from pydantic import BaseModel
from typing import Any, Dict, List

class TimeSeriesData(BaseModel):
    data: List[Dict[str, Any]]
    date_column: str
    value_column: str

class ForecastRequest(BaseModel):
    model_choice: str
    model_size: str = "small"
    forecast_horizon: int