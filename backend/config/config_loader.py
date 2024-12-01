"""
Configuration loader for the Time Series Forecasting Application.

This module handles loading and parsing of YAML configuration files.
"""

import os
from typing import Dict, Any
import yaml
import logging

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
            
            # Load visualization config
            vis_config_path = os.path.join(config_dir, 'visualization.yaml')
            with open(vis_config_path, 'r') as f:
                self._config = yaml.safe_load(f)
                
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise

    @property
    def model_colors(self) -> Dict[str, str]:
        return self._config.get('model_colors', {})

    @property
    def line_styles(self) -> Dict[str, Dict[str, Any]]:
        return self._config.get('line_styles', {})

    @property
    def plot_layout(self) -> Dict[str, Any]:
        return self._config.get('plot_layout', {})

    @property
    def metrics_table_style(self) -> Dict[str, str]:
        return self._config.get('metrics_table_style', {})

    @property
    def metrics_explanations(self) -> Dict[str, str]:
        return self._config.get('metrics_explanations', {})

# Create a singleton instance
config = ConfigLoader() 