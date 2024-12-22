# Handling missing values
# Check for missing values in the dataset
data.isnull().sum()

# For simplicity, let's fill missing values with the median for numerical columns
# and the most frequent value for categorical columns
data.fillna(data.median(numeric_only=True), inplace=True)

# Fill missing categorical values with the mode
data.fillna(data.select_dtypes(include='object').mode().iloc[0], inplace=True)

# Check if there are still any missing values
data.isnull().sum()