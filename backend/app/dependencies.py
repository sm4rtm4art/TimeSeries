from functools import lru_cache
from typing import Dict, Any

from fastapi import Depends, HTTPException, status
from darts import TimeSeries

from .config import Settings
from backend.data.loader import DataLoader
from backend.utils.session_state import initialize_session_state

@lru_cache()
def get_settings():
    return Settings()

def get_data_loader(settings: Settings = Depends(get_settings)):
    return DataLoader(settings)

async def get_time_series_data(
    data_loader: DataLoader = Depends(get_data_loader),
    settings: Settings = Depends(get_settings)
) -> TimeSeries:
    # This could be expanded to handle different data sources
    data = data_loader.load_data(settings.default_dataset)
    if data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Unable to load time series data"
        )
    return data

def get_model_config(settings: Settings = Depends(get_settings)) -> Dict[str, Any]:
    return {
        "available_models": settings.available_models,
        "default_model": settings.default_model,
        "model_sizes": settings.model_sizes,
        "default_model_size": settings.default_model_size,
    }

def initialize_streamlit_state(settings: Settings = Depends(get_settings)):
    initialize_session_state()
    # You could add more Streamlit-specific initializations here

# You can add more dependency functions as needed

