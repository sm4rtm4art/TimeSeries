"""Data loading"""

import pandas as pd
import streamlit as st
from darts import TimeSeries
from darts.datasets import AirPassengersDataset, ElectricityConsumptionZurichDataset, MonthlyMilkIncompleteDataset

from backend.utils.data_handling import prepare_data


class DataLoader:
    def __init__(self):
        pass

    def load_data(self):
        st.subheader("Data Loading")
        data_option = st.radio(
            "Choose data source:",
            [
                "Air Passengers",
                "Monthly Milk Production (Incomplete)",
                "Electricity Consumption (Zurich)",
                "Upload CSV",
            ],
        )

        if data_option == "Upload CSV":
            return self._load_csv()
        elif data_option == "Air Passengers":
            return self._load_air_passengers()
        elif data_option == "Monthly Milk Production (Incomplete)":
            return self._load_monthly_milk()
        else:  # Electricity Consumption (Zurich)
            return self._load_electricity_consumption()

    def _load_csv(self):
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file, parse_dates=["date"], index_col="date")
            time_series = TimeSeries.from_dataframe(data)
            st.success("Data loaded successfully!")
            return self._process_data(time_series)
        else:
            st.info("Please upload a CSV file.")
            return None, None, None

    def _load_air_passengers(self):
        time_series = AirPassengersDataset().load()
        st.success("Air Passengers dataset loaded successfully!")
        return self._process_data(time_series)

    def _load_monthly_milk(self):
        time_series = MonthlyMilkIncompleteDataset().load()
        st.success("Monthly Milk Production (Incomplete) dataset loaded successfully!")
        return self._process_data(time_series)

    def _load_electricity_consumption(self):
        time_series = ElectricityConsumptionZurichDataset().load()
        st.success("Electricity Consumption (Zurich) dataset loaded successfully!")
        return self._process_data(time_series)

    def _process_data(self, time_series):
        # Display dataset information
        st.subheader("Dataset Information")
        st.write(f"Start date: {time_series.start_time()}")
        st.write(f"End date: {time_series.end_time()}")
        st.write(f"Frequency: {time_series.freq}")
        st.write(f"Number of data points: {len(time_series)}")

        # Prepare train/test split
        train_data, test_data = prepare_data(time_series)

        return time_series, train_data, test_data
