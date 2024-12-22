
import pandas as pd

def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset by filling numerical columns with the median
    and categorical columns with the mode.

    Parameters:
    data (pd.DataFrame): The input data with potential missing values.

    Returns:
    pd.DataFrame: The data with missing values handled.
    """
    # Fill missing numerical values with the median
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())

    # Fill missing categorical values with the mode
    categorical_cols = data.select_dtypes(include=['object']).columns
    data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])

    return data
