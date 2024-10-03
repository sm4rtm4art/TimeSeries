import streamlit as st


def initialize_session_state():
    if 'forecast_horizon' not in st.session_state:
        st.session_state.forecast_horizon = 12  # or any default value you prefer
    default_values = {
        'data': None,
        'train_data': None,
        'test_data': None,
        'trained_models': {},
        'forecasts': {},
        'is_trained': False,
        'is_forecast_generated': False,
        'forecast_button': False,  # Add this line
        'forecast_horizon': 7  # Add this line with a default value
    }
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value
