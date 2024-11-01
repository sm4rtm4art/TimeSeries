import traceback
import logging
from typing import Dict, Union
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.metrics import mae, mape, mse, rmse, smape

logger = logging.getLogger(__name__)

def calculate_metrics(actual: TimeSeries, predicted: TimeSeries) -> Dict[str, float]:
    """Calculate standardized metrics for all models."""
    try:
        # Ensure series are aligned
        common_start = max(actual.start_time(), predicted.start_time())
        common_end = min(actual.end_time(), predicted.end_time())
        
        actual_trimmed = actual.slice(common_start, common_end)
        predicted_trimmed = predicted.slice(common_start, common_end)
        
        logger.info(f"Calculating metrics between series of lengths: actual={len(actual_trimmed)}, predicted={len(predicted_trimmed)}")
        
        return {
            'MAPE': float(mape(actual_trimmed, predicted_trimmed)),
            'RMSE': float(rmse(actual_trimmed, predicted_trimmed)),
            'MSE': float(mse(actual_trimmed, predicted_trimmed)),
            'MAE': float(mae(actual_trimmed, predicted_trimmed)),
            'sMAPE': float(smape(actual_trimmed, predicted_trimmed))
        }
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'MAPE': float('nan'),
            'RMSE': float('nan'),
            'MSE': float('nan'),
            'MAE': float('nan'),
            'sMAPE': float('nan')
        }

def format_metrics(metrics: Dict[str, float]) -> Dict[str, str]:
    """Format metrics for display."""
    try:
        return {
            'MAE': f"{metrics['MAE']:.2f}",
            'MSE': f"{metrics['MSE']:.2f}",
            'RMSE': f"{metrics['RMSE']:.2f}",
            'MAPE': f"{metrics['MAPE']:.2f}%",
            'sMAPE': f"{metrics['sMAPE']:.2f}%"
        }
    except Exception as e:
        logger.error(f"Error formatting metrics: {str(e)}")
        return {k: 'N/A' for k in ['MAE', 'MSE', 'RMSE', 'MAPE', 'sMAPE']}
