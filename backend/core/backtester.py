from typing import Dict, Union
from darts import TimeSeries
from ..utils.metrics import calculate_metrics

class ModelBacktester:
    @staticmethod
    def backtest(
        model: BaseModel,
        data: TimeSeries,
        start: Union[pd.Timestamp, float],
        forecast_horizon: int,
        stride: int = 1
    ) -> Dict[str, Union[TimeSeries, Dict[str, float]]]:
        """Generic backtesting pipeline."""
        historical_forecasts = model.historical_forecasts(
            series=data,
            start=start,
            forecast_horizon=forecast_horizon,
            stride=stride
        )
        
        metrics = calculate_metrics(data, historical_forecasts)
        
        return {
            'backtest': historical_forecasts,
            'metrics': metrics
        } 