"""Functionality for initializing and managing Streamlit session state.

It contains utilities to set up default values for various application variables,
ensuring consistent state management across the Streamlit application.
"""

from typing import Any

import streamlit as st


def initialize_session_state() -> None:
    """Initialize session state variables."""
    if "data" not in st.session_state:
        st.session_state.data = None
    if "train_data" not in st.session_state:
        st.session_state.train_data = None
    if "test_data" not in st.session_state:
        st.session_state.test_data = None
    if "trained_models" not in st.session_state:
        st.session_state.trained_models = None
    if "forecasts" not in st.session_state:
        st.session_state.forecasts = None
    if "backtests" not in st.session_state:
        st.session_state.backtests = None
    if "last_data_option" not in st.session_state:
        st.session_state.last_data_option = None


def get_session_state() -> dict[str, Any]:
    """Get the current Streamlit session state.

    Returns:
        Dictionary-like object containing all session state variables
    """
    return st.session_state
