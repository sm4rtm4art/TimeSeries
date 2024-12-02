from typing import Dict, Union
import pandas as pd
from darts import TimeSeries
from darts.metrics import mape, rmse, mae
import logging
from .interfaces.base_model import BasePredictor

logger = logging.getLogger(__name__)

class ModelBacktester:
    @staticmethod
    def generate_backtests(
        models: Dict[str, BasePredictor],
        data: TimeSeries,
        test_data: TimeSeries,
        forecast_horizon: int
    ) -> Dict[str, Dict[str, Union[TimeSeries, Dict[str, float]]]]:
        """Generate backtests for multiple models."""
        backtests = {}
        
        # Calculate start point for backtesting (beginning of test data)
        start = test_data.start_time()
        
        for model_name, model in models.items():
            try:
                logger.info(f"Generating backtest for {model_name}")
                backtest_result = ModelBacktester.backtest(
                    model=model,
                    data=data,
                    start=start,
                    forecast_horizon=forecast_horizon
                )
                
                backtests[model_name] = backtest_result
                logger.info(f"Successfully generated backtest for {model_name}")
                
            except Exception as e:
                logger.error(f"Error generating backtest for {model_name}: {str(e)}")
                continue
        
        return backtests

    @staticmethod
    def backtest(
        model: BasePredictor,
        data: TimeSeries,
        start: pd.Timestamp,
        forecast_horizon: int,
        stride: int = 1
    ) -> Dict[str, Union[TimeSeries, Dict[str, float]]]:
        """Generate backtest for a single model."""
        try:
            # Generate historical forecasts
            historical_forecasts = model.historical_forecasts(
                series=data,
                start=start,
                forecast_horizon=forecast_horizon,
                stride=stride,
                retrain=False,  # Set to True if you want to retrain at each step
                verbose=True
            )
            
            # Calculate metrics using actual test data
            actual_values = data.slice(start)
            metrics = {
                'MAPE': mape(actual_values, historical_forecasts),
                'RMSE': rmse(actual_values, historical_forecasts),
                'MAE': mae(actual_values, historical_forecasts)
            }
            
            logger.info(f"Backtest metrics: {metrics}")
            
            return {
                'backtest': historical_forecasts,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}")
            raise 