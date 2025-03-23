"""Functionality for initializing and managing Streamlit session state.

It contains utilities to set up default values for various application variables,
ensuring consistent state management across the Streamlit application.
"""

from typing import Any, TypedDict, cast

import streamlit as st
from darts import TimeSeries


class SessionState(TypedDict, total=False):
    """Type definition for Streamlit session state variables."""

    # Data variables
    data: TimeSeries | None
    train_data: TimeSeries | None
    test_data: TimeSeries | None
    last_data_option: str | None

    # Model variables
    trained_models: dict[str, Any] | None
    model_choice: str | None
    model_size: str
    training_started: bool

    # Forecast variables
    forecasts: dict[str, dict[str, TimeSeries]] | None
    backtests: dict[str, dict[str, Any]] | None


class SessionStateManager:
    """Wrapper class for Streamlit session state with proper type annotations."""

    def __init__(self) -> None:
        """Initialize session state variables if they don't exist."""
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
        if "training_started" not in st.session_state:
            st.session_state.training_started = False
        if "model_choice" not in st.session_state:
            st.session_state.model_choice = None
        if "model_size" not in st.session_state:
            st.session_state.model_size = "small"

    @property
    def data(self) -> TimeSeries | None:
        """Get the main time series data."""
        return st.session_state.data

    @data.setter
    def data(self, value: TimeSeries | None) -> None:
        st.session_state.data = value

    @property
    def train_data(self) -> TimeSeries | None:
        """Get the training time series data."""
        return st.session_state.train_data

    @train_data.setter
    def train_data(self, value: TimeSeries | None) -> None:
        st.session_state.train_data = value

    @property
    def test_data(self) -> TimeSeries | None:
        """Get the test time series data."""
        return st.session_state.test_data

    @test_data.setter
    def test_data(self, value: TimeSeries | None) -> None:
        st.session_state.test_data = value

    @property
    def last_data_option(self) -> str | None:
        """Get the last selected data option."""
        return cast(str | None, st.session_state.last_data_option)

    @last_data_option.setter
    def last_data_option(self, value: str | None) -> None:
        st.session_state.last_data_option = value

    @property
    def trained_models(self) -> dict[str, Any] | None:
        """Get the trained forecasting models."""
        return cast(dict[str, Any] | None, st.session_state.trained_models)

    @trained_models.setter
    def trained_models(self, value: dict[str, Any] | None) -> None:
        st.session_state.trained_models = value

    @property
    def forecasts(self) -> dict[str, dict[str, TimeSeries]] | None:
        """Get the forecasting results."""
        return cast(dict[str, dict[str, TimeSeries]] | None, st.session_state.forecasts)

    @forecasts.setter
    def forecasts(self, value: dict[str, dict[str, TimeSeries]] | None) -> None:
        st.session_state.forecasts = value

    @property
    def backtests(self) -> dict[str, dict[str, Any]] | None:
        """Get the backtesting results."""
        return cast(dict[str, dict[str, Any]] | None, st.session_state.backtests)

    @backtests.setter
    def backtests(self, value: dict[str, dict[str, Any]] | None) -> None:
        st.session_state.backtests = value

    @property
    def model_choice(self) -> str | None:
        """Get the selected model choice."""
        return cast(str | None, st.session_state.model_choice)

    @model_choice.setter
    def model_choice(self, value: str | None) -> None:
        st.session_state.model_choice = value

    @property
    def model_size(self) -> str:
        """Get the selected model size."""
        return cast(str, st.session_state.model_size)

    @model_size.setter
    def model_size(self, value: str) -> None:
        st.session_state.model_size = value

    @property
    def training_started(self) -> bool:
        """Get the training started flag."""
        return cast(bool, st.session_state.training_started)

    @training_started.setter
    def training_started(self, value: bool) -> None:
        st.session_state.training_started = value


def initialize_session_state() -> None:
    """Initialize session state variables."""
    # Create a session state manager which will initialize all variables
    SessionStateManager()


def get_session_state() -> SessionStateManager:
    """Get the session state manager with type annotations.

    Returns:
        SessionStateManager: A wrapper around the Streamlit session state
    """
    return SessionStateManager()
