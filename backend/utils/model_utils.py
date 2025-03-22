import numpy as np
import torch


def get_training_config() -> dict:
    """Determines the appropriate accelerator and precision settings based on available hardware.
    For MPS (Apple Silicon), we need to ensure data is in float32.
    For other devices, we use float64 for better precision.
    """
    if torch.cuda.is_available():
        return {
            "accelerator": "gpu",
            "precision": "64-true",
            "force_dtype": np.float64,
        }
    elif torch.backends.mps.is_available():
        return {
            "accelerator": "mps",
            "precision": "32-true",
            "force_dtype": np.float32,
        }
    else:
        return {
            "accelerator": "cpu",
            "precision": "64-true",
            "force_dtype": np.float64,
        }
