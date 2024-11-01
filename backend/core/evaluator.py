from typing import Dict, Union
from darts import TimeSeries
from darts.metrics import mape, rmse, mse, mae, smape
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    @staticmethod
    def calculate_metrics(
        actual: TimeSeries,
        predicted: TimeSeries
    ) -> Dict[str, float]:
        """Calculate standard metrics for model evaluation"""
        try:
            # Ensure series are aligned
            common_start = max(actual.start_time(), predicted.start_time())
            common_end = min(actual.end_time(), predicted.end_time())
            
            actual_trimmed = actual.slice(common_start, common_end)
            predicted_trimmed = predicted.slice(common_start, common_end)
            
            return {
                'MAPE': float(mape(actual_trimmed, predicted_trimmed)),
                'RMSE': float(rmse(actual_trimmed, predicted_trimmed)),
                'MSE': float(mse(actual_trimmed, predicted_trimmed)),
                'MAE': float(mae(actual_trimmed, predicted_trimmed)),
                'sMAPE': float(smape(actual_trimmed, predicted_trimmed))
            }
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {metric: float('nan') for metric in ['MAPE', 'RMSE', 'MSE', 'MAE', 'sMAPE']}

    @staticmethod
    def evaluate_models(
        actual: TimeSeries,
        forecasts: Dict[str, Dict[str, TimeSeries]]
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate multiple models"""
        return {
            model_name: ModelEvaluator.calculate_metrics(actual, forecast_dict['future'])
            for model_name, forecast_dict in forecasts.items()
            if 'future' in forecast_dict
        } 