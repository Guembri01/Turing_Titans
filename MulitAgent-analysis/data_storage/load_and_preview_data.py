import pandas as pd

# Load the dataset
file_path = 'OnlineSalesData.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
data.head()