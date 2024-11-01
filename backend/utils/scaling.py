"""
Scaling utilities for time series data
"""

from typing import Tuple
import numpy as np
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

def scale_data(data: TimeSeries) -> Tuple[TimeSeries, Scaler]:
    """
    Scale the time series data using Darts' Scaler.
    
    Args:
        data: Input TimeSeries data
        
    Returns:
        Tuple containing:
        - Scaled TimeSeries
        - Fitted Scaler object
    """
    try:
        # Ensure data is float32
        data = data.astype(np.float32)
        
        # Initialize scaler
        scaler = Scaler()
        
        # Fit and transform data
        scaled_data = scaler.fit_transform(data)
        
        return scaled_data, scaler
        
    except Exception as e:
        raise ValueError(f"Error scaling data: {str(e)}")

def inverse_scale(data: TimeSeries, scaler: Scaler) -> TimeSeries:
    """
    Inverse scale the time series data.
    
    Args:
        data: Scaled TimeSeries data
        scaler: Fitted Scaler object
        
    Returns:
        Original-scale TimeSeries
    """
    try:
        # Ensure data is float32
        data = data.astype(np.float32)
        
        # Inverse transform
        return scaler.inverse_transform(data)
        
    except Exception as e:
        raise ValueError(f"Error inverse scaling data: {str(e)}")
