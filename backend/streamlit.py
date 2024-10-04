import streamlit as st
from backend.data.data_loader import load_data
from backend.utils.app_components import display_results, generate_forecasts, train_models
from backend.utils.data_handling import prepare_data
from backend.utils.session_state import initialize_session_state
from backend.utils.ui_components import display_sidebar
from backend.utils.metrics import calculate_metrics

import traceback

def main():
    initialize_session_state()
    
    model_choice, model_size, forecast_horizon, train_button = display_sidebar()
    
    st.title("Time Series Forecasting App")
    
    # Load data
    data = load_data()
    if data is not None:
        st.session_state.data = data
        st.session_state.train_data, st.session_state.test_data = prepare_data(data)
        
        # Train models
        if train_button:
            with st.spinner("Training models... This may take a few minutes."):
                try:
                    st.session_state.trained_models, st.session_state.backtests = train_models(
                        st.session_state.train_data,
                        st.session_state.test_data,
                        model_choice,
                        model_size
                    )
                    st.session_state.is_trained = True
                    st.success("Models trained successfully!")
                except Exception as e:
                    print("An error occurred during model training:")
                    print(traceback.format_exc())
                    st.error(f"An error occurred during model training: {str(e)}")
        
        # Generate forecasts
        if st.session_state.is_trained:
            if st.button("Generate Forecasts"):
                with st.spinner("Generating forecasts..."):
                    try:
                        st.session_state.forecasts = generate_forecasts(
                            st.session_state.trained_models,
                            st.session_state.data,
                            st.session_state.test_data,
                            forecast_horizon,
                            st.session_state.backtests
                        )
                        st.session_state.is_forecast_generated = True
                        st.success("Forecasts generated successfully!")
                    except Exception as e:
                        print("An error occurred during forecast generation:")
                        print(traceback.format_exc())
                        st.error(f"An error occurred during forecast generation: {str(e)}")
        
        # Display results
        if st.session_state.is_forecast_generated:
            try:
                display_results(
                    st.session_state.data,
                    st.session_state.test_data,
                    st.session_state.forecasts,
                    model_choice,
                    forecast_horizon
                )
            except Exception as e:
                print("An error occurred while displaying results:")
                print(traceback.format_exc())
                st.error(f"An error occurred while displaying results: {str(e)}")

if __name__ == "__main__":
    main()