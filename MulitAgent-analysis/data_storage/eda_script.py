import pandas as pd

# Load the processed dataset
data = pd.read_csv('Processed_OnlineSalesData.csv')

# Generate descriptive statistics
descriptive_stats = data.describe()

# Calculate the correlation matrix
correlation_matrix = data.corr()

# Display results
descriptive_stats, correlation_matrix