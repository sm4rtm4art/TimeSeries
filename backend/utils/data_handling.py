import streamlit as st
from typing import Tuple
from darts import TimeSeries
import logging
import pandas as pd
import numpy as np
import traceback

logger = logging.getLogger(__name__)

def prepare_data(data: TimeSeries, test_size: float = 0.2) -> Tuple[TimeSeries, TimeSeries]:
    """Split time series data into train and test sets."""
    try:
        logger.info(f"Preparing data split with test_size: {test_size}")
        logger.info(f"Input data type: {type(data)}")
        logger.info(f"Input data length: {len(data)}")
        
        train_size = int(len(data) * (1 - test_size))
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        logger.info(f"Split complete - Train size: {len(train_data)}, Test size: {len(test_data)}")
        return train_data, test_data
        
    except Exception as e:
        logger.error(f"Error in prepare_data: {str(e)}")
        logger.error(traceback.format_exc())
        raise

class DataHandler:
    @staticmethod
    def load_data() -> TimeSeries:
        try:
            # Create sample dataset
            dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
            values = np.random.normal(loc=100, scale=10, size=len(dates))
            df = pd.DataFrame({'date': dates, 'value': values})
            
            # Convert to TimeSeries and add debug logging
            series = TimeSeries.from_dataframe(df, 'date', 'value')
            logger.info(f"Created TimeSeries with {len(series)} timestamps")
            logger.info(f"TimeSeries type: {type(series)}")
            return series
            
        except Exception as e:
            logger.error(f"Error in load_data: {str(e)}")
            logger.error(traceback.format_exc())
            raise
