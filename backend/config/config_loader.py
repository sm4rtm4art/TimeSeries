import logging
import os
from typing import Any

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


class ConfigLoader:
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self._load_config()

    def _load_config(self) -> None:
        """Load configuration from YAML files."""
        try:
            config_dir = os.path.dirname(os.path.abspath(__file__))

            # Load all configs
            vis_config_path = os.path.join(config_dir, "visualization.yaml")
            model_config_path = os.path.join(config_dir, "model_config.yaml")
            hardware_config_path = os.path.join(config_dir, "hardware_config.yaml")

            self._config = {}

            # Load configs
            with open(vis_config_path) as f:
                self._config["visualization"] = yaml.safe_load(f)
            with open(model_config_path) as f:
                self._config["model"] = yaml.safe_load(f)
            with open(hardware_config_path) as f:
                self._config["hardware"] = yaml.safe_load(f)

            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise

    def get_model_config(self, model_name: str) -> dict[str, Any]:
        """Get model-specific configuration."""
        return self._config.get("model", {}).get("models", {}).get(model_name, {})

    def get_hardware_config(self) -> dict[str, Any]:
        """Get hardware-specific configuration with proper MPS handling."""
        if torch.cuda.is_available():
            return {
                "accelerator": "gpu",
                "precision": self._config["hardware"]["accelerator"]["cuda"]["precision"],
                "force_dtype": np.float64,
                "enabled": True,
            }
        elif torch.backends.mps.is_available():
            return {
                "accelerator": "mps",  # Keep MPS as accelerator
                "precision": "32-true",
                "force_dtype": np.float32,
                "enabled": True,
            }
        else:
            return {
                "accelerator": "cpu",
                "precision": self._config["hardware"]["accelerator"]["cpu"]["precision"],
                "force_dtype": np.float64,
                "enabled": True,
            }

    @property
    def model_colors(self) -> dict[str, str]:
        return self._config.get("visualization", {}).get("model_colors", {})

    @property
    def line_styles(self) -> dict[str, dict[str, Any]]:
        return self._config.get("visualization", {}).get("line_styles", {})

    @property
    def plot_layout(self) -> dict[str, Any]:
        return self._config.get("visualization", {}).get("plot_layout", {})

    @property
    def metrics_table_style(self) -> dict[str, str]:
        return self._config.get("visualization", {}).get("metrics_table_style", {})

    @property
    def metrics_explanations(self) -> dict[str, str]:
        return self._config.get("visualization", {}).get("metrics_explanations", {})


# Create a singleton instance
config = ConfigLoader()
