import streamlit as st
from backend.data.data_loader import DataLoader
from backend.data.data_preprocessing import detect_outliers
from backend.utils.app_components import display_results, generate_forecasts, train_models
from backend.utils.session_state import initialize_session_state
from backend.utils.ui_components import display_sidebar
from backend.utils.plotting import TimeSeriesPlotter

def main():
    initialize_session_state()
    
    st.title("Time Series Forecasting App")
    
    model_choice, model_size, train_button, forecast_horizon, forecast_button = display_sidebar()
    
    data_loader = DataLoader()
    data, train_data, test_data = data_loader.load_data()
    
    if data is not None:
        st.session_state.data = data
        st.session_state.train_data = train_data
        st.session_state.test_data = test_data
        
        st.subheader("Data Preprocessing")
        
        # Outlier detection options
        st.write("Outlier Detection")
        outlier_method = st.selectbox("Select outlier detection method", 
                                      ['combined', 'knn', 'iforest', 'lof'])
        contamination = st.slider("Contamination (proportion of outliers)", 0.01, 0.1, 0.01, 0.01)
        n_neighbors = st.slider("Number of neighbors (for KNN and LOF)", 5, 50, 20)
        
        if st.button("Detect Outliers"):
            outliers = detect_outliers(data, method=outlier_method, 
                                       contamination=contamination, n_neighbors=n_neighbors)
            
            # Visualize outliers
            plotter = TimeSeriesPlotter()
            fig = plotter.plot_outliers(data, outliers)
            st.plotly_chart(fig)
            
            st.write(f"Number of outliers detected: {outliers.sum().sum()}")
        
        # Display original data
        st.subheader("Original Data")
        st.line_chart(data.pd_dataframe())

        # Display train/test split
        st.subheader("Train/Test Split")
        split_data = train_data.pd_dataframe()
        split_data["Test"] = test_data.pd_dataframe()
        st.line_chart(split_data)

        # Train models
        if train_button:
            with st.spinner("Training models... This may take a few minutes."):
                try:
                    st.session_state.trained_models, st.session_state.backtests = train_models(
                        st.session_state.train_data, st.session_state.test_data, model_choice, model_size
                    )
                    st.success("Models trained successfully!")
                except Exception as e:
                    print("An error occurred during model training:")
                    print(traceback.format_exc())
                    st.error(f"An error occurred during model training: {str(e)}")

        # Generate forecast
        if st.session_state.trained_models and forecast_button:
            with st.spinner("Generating forecasts..."):
                try:
                    st.session_state.forecasts = generate_forecasts(
                        st.session_state.trained_models,
                        st.session_state.data,
                        forecast_horizon,
                        st.session_state.backtests,
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
                    st.session_state.train_data,
                    st.session_state.test_data,
                    st.session_state.forecasts,
                    model_choice,
                    forecast_horizon,
                )
            except Exception as e:
                print("An error occurred while displaying results:")
                print(traceback.format_exc())
                st.error(f"An error occurred while displaying results: {str(e)}")


if __name__ == "__main__":
    main()
