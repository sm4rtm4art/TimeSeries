from typing import Dict, Any
from darts import TimeSeries
import streamlit as st
from ...core.model_factory import train_models

def train_and_store_models(
    train_data: TimeSeries,
    model_choice: str = "All Models",
    model_size: str = "small"
) -> Dict[str, Any]:
    """Wrapper function to train and store models in session state"""
    trained_models = train_models(train_data, model_choice, model_size)
    st.session_state.trained_models = trained_models
    return trained_models