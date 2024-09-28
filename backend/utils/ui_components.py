import streamlit as st

def display_sidebar():
    with st.sidebar:
        st.header("Model Settings")
        st.session_state.model_choice = st.selectbox("Choose Model", ["All Models", "N-BEATS", "Prophet", "TiDE", "Chronos"])
        if st.session_state.model_choice in ["All Models", "Chronos"]:
            st.session_state.model_size = st.selectbox("Chronos Model Size", ["tiny", "small", "medium", "large"])
        st.session_state.forecast_horizon = st.slider("Forecast Horizon (periods)", min_value=1, max_value=36, value=12)
        st.session_state.train_button = st.button("Train Models")
        st.session_state.forecast_button = st.button("Generate Forecast")