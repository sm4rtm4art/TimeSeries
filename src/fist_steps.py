"""
First steps with NumPy and Pandas.
This module demonstrates a simple function that utilizes both libraries and includes a test.
"""

import numpy as np
import pandas as pd


def generate_random_dataframe(rows: int, cols: int) -> pd.DataFrame:
    """
    Generate a DataFrame with random numbers using NumPy.

    Args:
        rows (int): Number of rows in the DataFrame.
        cols (int): Number of columns in the DataFrame.

    Returns:
        pd.DataFrame: A DataFrame filled with random numbers.
    """
    data = np.random.random(size=(rows, cols))
    return pd.DataFrame(data, columns=[f"Column_{i}" for i in range(cols)])


def describe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return descriptive statistics for the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: Descriptive statistics of the DataFrame.
    """
    return df.describe()


def main() -> None:
    """
    Main function that generates a DataFrame and prints its description.
    """
    df = generate_random_dataframe(10, 5)
    print("Generated DataFrame:")
    print(df)
    print("\nDescriptive Statistics:")
    print(describe_dataframe(df))


if __name__ == "__main__":
    main()
