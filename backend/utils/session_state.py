"""
This module provides functionality for initializing and managing Streamlit session state.

It contains utilities to set up default values for various application variables,
ensuring consistent state management across the Streamlit application.
"""

import streamlit as st

def initialize_session_state():
    """Initialize session state variables"""
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'train_data' not in st.session_state:
        st.session_state.train_data = None
    if 'test_data' not in st.session_state:
        st.session_state.test_data = None
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    if 'forecasts' not in st.session_state:
        st.session_state.forecasts = {}
    if 'backtests' not in st.session_state:
        st.session_state.backtests = {}
    if 'model_metrics' not in st.session_state:
        st.session_state.model_metrics = {}
    if 'is_forecast_generated' not in st.session_state:
        st.session_state.is_forecast_generated = False
    if 'model_choice' not in st.session_state:
        st.session_state.model_choice = "All Models"
    if 'model_size' not in st.session_state:
        st.session_state.model_size = "small"
    if 'forecast_horizon' not in st.session_state:
        st.session_state.forecast_horizon = 30

def get_session_state():
    return st.session_state
