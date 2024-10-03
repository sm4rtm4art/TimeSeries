import streamlit as st

# import logging



# logger = logging.getLogger(__name__)


def load_data_if_needed():
    if st.session_state.data is None:
        try:
            # logger.info("Loading data...")
            st.session_state.data = load_data()
            # logger.info("Data loaded successfully")
            st.success("Data loaded successfully")
        except Exception as e:
            #logger.error(f"Failed to load data: {type(e).__name__}: {str(e)}")
            st.error(f"Failed to load data: {type(e).__name__}: {str(e)}")
            st.stop()


def prepare_data(data, test_size=0.2):
    # Split the data into training and testing sets
    train_size = int(len(data) * (1 - test_size))
    train_data = data[:train_size]
    test_data = data[train_size:]

    return train_data, test_data
