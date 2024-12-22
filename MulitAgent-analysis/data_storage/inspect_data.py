import pandas as pd

# Load the processed data to inspect column names
processed_data = pd.read_csv('OnlineSalesData.csv')

# Display the first few rows to identify potential target columns
processed_data.head()