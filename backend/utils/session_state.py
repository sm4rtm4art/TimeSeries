import streamlit as st


def initialize_session_state():
    default_values = {
        'data': None,
        'train_data': None,
        'test_data': None,
        'trained_models': {},
        'forecasts': {},
        'is_trained': False,
        'is_forecast_generated': False
    }
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value