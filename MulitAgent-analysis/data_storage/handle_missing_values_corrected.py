import pandas as pd

# Reload the dataset
file_path = 'OnlineSalesData.csv'
data = pd.read_csv(file_path)

# Handling missing values
# Check for missing values in the dataset
missing_values = data.isnull().sum()

# Fill missing numerical values with the median
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())

# Fill missing categorical values with the mode
categorical_cols = data.select_dtypes(include=['object']).columns
data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])

# Check if there are still any missing values after filling
remaining_missing_values = data.isnull().sum()

missing_values, remaining_missing_values