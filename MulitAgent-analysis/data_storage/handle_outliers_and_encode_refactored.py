
import pandas as pd

def handle_outliers_and_encode(data: pd.DataFrame) -> pd.DataFrame:
    """
    Handle outliers in numerical columns using the IQR method and encode categorical variables
    using one-hot encoding.

    Parameters:
    data (pd.DataFrame): The input data with numerical and categorical columns.

    Returns:
    pd.DataFrame: The data with outliers handled and categorical variables encoded.
    """
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns

    # Handling outliers using IQR method
    def cap_outliers(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return series.clip(lower_bound, upper_bound)

    # Apply the capping function to numerical columns
    data[numerical_cols] = data[numerical_cols].apply(cap_outliers)

    # Encoding categorical variables using one-hot encoding
    data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    return data_encoded
