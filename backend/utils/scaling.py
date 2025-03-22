"""Scaling utilities for time series data."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def scale_data(data: np.ndarray, method: str = "minmax") -> tuple[np.ndarray, dict]:
    """Scale data using the specified method.

    Args:
        data: Input data array to scale
        method: Scaling method ('minmax' or 'standard')

    Returns:
        Tuple containing scaled data and scaling parameters
    """
    try:
        if method == "minmax":
            # Min-max scaling
            data_min = np.min(data, axis=0)
            data_max = np.max(data, axis=0)
            scaled_data = (data - data_min) / (data_max - data_min + 1e-10)
            scaling_params = {"min": data_min, "max": data_max, "method": method}

        elif method == "standard":
            # Standard scaling (z-score normalization)
            data_mean = np.mean(data, axis=0)
            data_std = np.std(data, axis=0)
            scaled_data = (data - data_mean) / (data_std + 1e-10)
            scaling_params = {"mean": data_mean, "std": data_std, "method": method}

        else:
            raise ValueError(f"Unknown scaling method: {method}")

        return scaled_data, scaling_params
    except Exception as e:
        raise ValueError(f"Error scaling data: {str(e)}") from e


def inverse_scale_data(data: np.ndarray, scaling_params: dict) -> np.ndarray:
    """Convert scaled data back to original scale.

    Args:
        data: Scaled data array
        scaling_params: Dictionary with scaling parameters

    Returns:
        Unscaled data in original scale
    """
    try:
        method = scaling_params.get("method", "minmax")

        if method == "minmax":
            data_min = scaling_params["min"]
            data_max = scaling_params["max"]
            original_data = data * (data_max - data_min) + data_min

        elif method == "standard":
            data_mean = scaling_params["mean"]
            data_std = scaling_params["std"]
            original_data = data * data_std + data_mean

        else:
            raise ValueError(f"Unknown scaling method: {method}")

        return original_data
    except Exception as e:
        raise ValueError(f"Error inverse scaling data: {str(e)}") from e
