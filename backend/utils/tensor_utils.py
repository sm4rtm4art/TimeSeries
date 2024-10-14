import numpy as np
import torch
from darts import TimeSeries


def ensure_float32(data):
    """
    Ensure the data is in float32 format, especially when using MPS.
    
    :param data: Input data (can be TimeSeries, numpy array, or torch tensor)
    :return: Data converted to float32 if necessary
    """
    if isinstance(data, TimeSeries):
        return data.astype(np.float32)
    elif isinstance(data, np.ndarray):
        return data.astype(np.float32)
    elif isinstance(data, torch.Tensor):
        return data.float()
    else:
        return data  # Return as-is if type is not recognized


def is_mps_available():
    """
    Check if MPS (Metal Performance Shaders) is available.
    
    :return: True if MPS is available, False otherwise
    """
    return torch.backends.mps.is_available()


def ensure_tensor_float32(data):
    """
    Ensure the data is in float32 format, especially when using MPS.
    
    :param data: Input data (can be TimeSeries, numpy array, or torch tensor)
    :return: Data converted to float32 if necessary
    """
    if isinstance(data, TimeSeries):
        return data.astype(np.float32)
    elif isinstance(data, np.ndarray):
        return data.astype(np.float32)
    elif isinstance(data, torch.Tensor):
        return data.float()
    else:
        return data  # Return as-is if type is not recognized
