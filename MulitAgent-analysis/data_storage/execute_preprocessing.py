# Execute the preprocessing script with the refactored functions to handle missing values, cap outliers, and encode categorical variables.

# Import necessary libraries
import pandas as pd
from handle_missing_values_refactored import handle_missing_values
from handle_outliers_and_encode_refactored import handle_outliers_and_encode

# Load the dataset
file_path = 'OnlineSalesData.csv'
data = pd.read_csv(file_path)

# Handle missing values
data = handle_missing_values(data)

# Handle outliers and encode categorical variables
data = handle_outliers_and_encode(data)

# Save the processed data to a new CSV file
data.to_csv('Processed_OnlineSalesData.csv', index=False)

# Print success message
print('Data preprocessing completed successfully. Processed data saved to Processed_OnlineSalesData.csv.')