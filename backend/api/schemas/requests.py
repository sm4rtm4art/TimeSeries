from typing import Any

from pydantic import BaseModel


class TimeSeriesData(BaseModel):
    data: list[dict[str, Any]]
    date_column: str
    value_column: str


class ForecastRequest(BaseModel):
    model_choice: str
    model_size: str = "small"
    forecast_horizon: int
