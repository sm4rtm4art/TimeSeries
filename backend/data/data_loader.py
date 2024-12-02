"""
Data loading functionality for time series forecasting
"""
import pandas as pd
import streamlit as st
from darts import TimeSeries
from darts.datasets import (
    AirPassengersDataset,
    ElectricityConsumptionZurichDataset,
    MonthlyMilkIncompleteDataset
)
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self):
        """Initialize DataLoader."""
        pass

    def load_data(self, data_option: str) -> Tuple[Optional[TimeSeries], Optional[TimeSeries], Optional[TimeSeries]]:
        """
        Load data based on user selection.
        
        Args:
            self: Instance of DataLoader
            data_option: String indicating which dataset to load
        """
        try:
            if data_option == "Upload CSV":
                return self._load_csv()
            elif data_option == "Air Passengers":
                return self._load_air_passengers()
            elif data_option == "Monthly Milk Production":
                return self._load_monthly_milk()
            elif data_option == "Electricity Consumption (Zurich)":
                return self._load_electricity()
            else:
                st.error("Please select a valid data source")
                return None, None, None
                
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            st.error(f"Error loading data: {str(e)}")
            return None, None, None

    def _load_csv(self) -> Tuple[Optional[TimeSeries], Optional[TimeSeries], Optional[TimeSeries]]:
        """Load data from CSV file."""
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            date_col = st.selectbox("Select date column", df.columns)
            value_col = st.selectbox("Select value column", df.columns)
            time_series = TimeSeries.from_dataframe(df, date_col, value_col)
            return self._process_data(time_series)
        return None, None, None

    def _load_air_passengers(self) -> Tuple[TimeSeries, TimeSeries, TimeSeries]:
        """Load Air Passengers dataset."""
        time_series = AirPassengersDataset().load()
        st.success("Air Passengers dataset loaded successfully!")
        return self._process_data(time_series)

    def _load_monthly_milk(self) -> Tuple[TimeSeries, TimeSeries, TimeSeries]:
        """Load Monthly Milk Production dataset."""
        time_series = MonthlyMilkIncompleteDataset().load()
        st.success("Monthly Milk Production dataset loaded successfully!")
        return self._process_data(time_series)

    def _load_electricity(self) -> Tuple[TimeSeries, TimeSeries, TimeSeries]:
        """Load Electricity Consumption dataset."""
        time_series = ElectricityConsumptionZurichDataset().load()
        st.success("Electricity Consumption dataset loaded successfully!")
        return self._process_data(time_series)

    def _process_data(self, time_series: TimeSeries) -> Tuple[TimeSeries, TimeSeries, TimeSeries]:
        """Process the loaded data and split into train/test sets."""
        try:
            train_size = int(len(time_series) * 0.8)
            train_data = time_series[:train_size]
            test_data = time_series[train_size:]
            return time_series, train_data, test_data
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            st.error(f"Error processing data: {str(e)}")
            return None, None, None
