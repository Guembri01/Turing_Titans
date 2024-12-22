import pandas as pd

# Load the processed data
processed_data = pd.read_csv('OnlineSalesData.csv')

# Display the columns of the dataset
columns = processed_data.columns.tolist()
columns