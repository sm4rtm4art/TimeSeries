import numpy as np
import pandas as pd

from src.first_steps import describe_dataframe, generate_random_dataframe


def test_generate_random_dataframe():
    """Test the DataFrame generation function."""
    df = generate_random_dataframe(10, 5)

    # Check if DataFrame has the correct shape
    assert df.shape == (10, 5)

    # Check if the result is a pandas DataFrame
    assert isinstance(df, pd.DataFrame)

    # Check if all values are between 0 and 1 (from NumPy's random function)
    assert np.all(df.values >= 0) and np.all(df.values <= 1)


def test_describe_dataframe():
    """Test the descriptive statistics function."""
    df = generate_random_dataframe(10, 5)
    description = describe_dataframe(df)

    # Check if the describe DataFrame has 8 rows (count, mean, std, etc.)
    assert description.shape[0] == 8

    # Check if the result is a pandas DataFrame
    assert isinstance(description, pd.DataFrame)
