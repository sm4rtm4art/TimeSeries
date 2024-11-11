import pandas as pd
from darts import TimeSeries
import logging

logger = logging.getLogger(__name__)

class TimeSeriesUtils:
    @staticmethod
    def get_timedelta(series: TimeSeries, periods: int) -> pd.Timedelta:
        """Calculate time delta for a given time series and number of periods."""
        if len(series) < 2:
            raise ValueError("Series must have at least two data points to determine frequency.")
        
        freq = pd.infer_freq(series.time_index)
        
        if freq is None:
            time_diff = series.time_index[-1] - series.time_index[0]
            avg_diff = time_diff / (len(series) - 1)
            return avg_diff * periods
        
        if freq in ['D', 'H', 'T', 'S']:
            return pd.Timedelta(periods, freq)
        elif freq in ['M', 'MS']:
            return pd.offsets.MonthEnd(periods)
        elif freq in ['Y', 'YS']:
            return pd.offsets.YearEnd(periods)
        elif freq == 'W':
            return pd.Timedelta(weeks=periods)
        else:
            time_diff = series.time_index[1] - series.time_index[0]
            return time_diff * periods 