import pandas as pd
import streamlit as st
from darts import TimeSeries
from darts.dataprocessing.transformers import MissingValuesFiller
from darts.datasets import AirPassengersDataset, ElectricityConsumptionZurichDataset, MonthlyMilkIncompleteDataset


def load_data():
    st.subheader("Data Loading")
    data_option = st.radio("Choose data source:", [
        "Air Passengers",
        "Monthly Milk Production (Incomplete)",
        "Electricity Consumption (Zurich)",
        "Upload CSV"
    ])

    if data_option == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file, parse_dates=['date'], index_col='date')
            time_series = TimeSeries.from_dataframe(data)
            st.success("Data loaded successfully!")
        else:
            st.info("Please upload a CSV file.")
            return None
    elif data_option == "Air Passengers":
        time_series = AirPassengersDataset().load()
        st.success("Air Passengers dataset loaded successfully!")
    elif data_option == "Monthly Milk Production (Incomplete)":
        time_series = MonthlyMilkIncompleteDataset().load()
        st.success("Monthly Milk Production (Incomplete) dataset loaded successfully!")
    else:  # Electricity Consumption (Zurich)
        time_series = ElectricityConsumptionZurichDataset().load()
        st.success("Electricity Consumption (Zurich) dataset loaded successfully!")

    if time_series is not None:
        # Check for missing values
        if time_series.pd_dataframe().isnull().values.any():
            st.warning("This dataset contains missing values. Applying interpolation.")
            filler = MissingValuesFiller()
            time_series = filler.transform(time_series)
            st.success("Missing values have been interpolated.")

        st.subheader("Original Data")
        st.line_chart(time_series.pd_dataframe())

        # Display dataset information
        st.subheader("Dataset Information")
        st.write(f"Start date: {time_series.start_time()}")
        st.write(f"End date: {time_series.end_time()}")
        st.write(f"Frequency: {time_series.freq}")
        st.write(f"Number of data points: {len(time_series)}")

    return time_series


def prepare_data(data, test_size=0.2):
    # Split the data into training and testing sets
    train_size = int(len(data) * (1 - test_size))
    train_data = data[:train_size]
    test_data = data[train_size:]

    return train_data, test_data
