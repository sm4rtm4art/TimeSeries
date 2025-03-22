"""Data loading functionality for time series forecasting"""

import logging

import pandas as pd
import streamlit as st
from darts import TimeSeries
from darts.datasets import AirPassengersDataset, ElectricityConsumptionZurichDataset, MonthlyMilkIncompleteDataset

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self):
        """Initialize DataLoader."""
        pass

    def load_data(self, data_option: str, test_size: float = 0.2) -> tuple[TimeSeries, TimeSeries, TimeSeries]:
        """Load and split data based on selected option."""
        try:
            if data_option == "Upload CSV":
                return self._load_csv()
            elif data_option == "Air Passengers":
                time_series = AirPassengersDataset().load()
                st.success("Air Passengers dataset loaded successfully!")
            elif data_option == "Monthly Milk Production":
                time_series = MonthlyMilkIncompleteDataset().load()
                st.success("Monthly Milk Production dataset loaded successfully!")
            else:  # Electricity Consumption
                time_series = ElectricityConsumptionZurichDataset().load()
                st.success("Electricity Consumption dataset loaded successfully!")

            # Process the data
            train_size = int(len(time_series) * (1 - test_size))
            train_data = time_series[:train_size]
            test_data = time_series[train_size:]
            return time_series, train_data, test_data

        except Exception as e:
            logger.error("Error loading data: %s", str(e))
            st.error(f"Error loading data: {str(e)}")
            return None, None, None

    def _load_csv(self) -> tuple[TimeSeries | None, TimeSeries | None, TimeSeries | None]:
        """Load data from CSV file."""
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            date_col = st.selectbox("Select date column", df.columns)
            value_col = st.selectbox("Select value column", df.columns)
            time_series = TimeSeries.from_dataframe(df, date_col, value_col)
            return self._process_data(time_series)
        return None, None, None

    def _load_air_passengers(self) -> tuple[TimeSeries, TimeSeries, TimeSeries]:
        """Load Air Passengers dataset."""
        time_series = AirPassengersDataset().load()
        st.success("Air Passengers dataset loaded successfully!")
        return self._process_data(time_series)

    def _load_monthly_milk(self) -> tuple[TimeSeries, TimeSeries, TimeSeries]:
        """Load Monthly Milk Production dataset."""
        time_series = MonthlyMilkIncompleteDataset().load()
        st.success("Monthly Milk Production dataset loaded successfully!")
        return self._process_data(time_series)

    def _load_electricity_consumption(self) -> tuple[TimeSeries, TimeSeries, TimeSeries]:
        """Load Electricity Consumption dataset."""
        time_series = ElectricityConsumptionZurichDataset().load()
        st.success("Electricity Consumption dataset loaded successfully!")
        return self._process_data(time_series)

    def _process_data(self, time_series: TimeSeries) -> tuple[TimeSeries, TimeSeries, TimeSeries]:
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
