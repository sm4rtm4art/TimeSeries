"""
This module provides functionality for initializing and managing Streamlit session state.

It contains utilities to set up default values for various application variables,
ensuring consistent state management across the Streamlit application.
"""

import streamlit as st

def initialize_session_state():
    """
    Initialize the Streamlit session state with default values for various application variables.

    This function sets up the initial state for the Streamlit application, ensuring that
    all necessary variables are present in the session state. If a variable already exists
    in the session state, its value is not overwritten.

    The function initializes the following variables:
    - data: Stores the main dataset (default: None)
    - train_data: Stores the training dataset (default: None)
    - test_data: Stores the testing dataset (default: None)
    - trained_models: Dictionary to store trained models (default: empty dict)
    - forecasts: Dictionary to store generated forecasts (default: empty dict)
    - is_trained: Flag indicating if models have been trained (default: False)
    - is_forecast_generated: Flag indicating if forecasts have been generated (default: False)
    - forecast_button: State of the forecast button (default: False)
    - forecast_horizon: Number of time steps to forecast (default: 7)

    Usage:
        Call this function at the beginning of your Streamlit app to ensure
        all necessary session state variables are initialized.
    """
    default_values = {
        'data': None,
        'train_data': None,
        'test_data': None,
        'trained_models': {},
        'forecasts': {},
        'is_trained': False,
        'is_forecast_generated': False,
        'forecast_button': False,
        'forecast_horizon': 7
    }

    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_session_state():
    return st.session_state
