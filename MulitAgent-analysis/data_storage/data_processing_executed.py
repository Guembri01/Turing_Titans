import pandas as pd


def load_and_process_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Drop any duplicate rows
    data.drop_duplicates(inplace=True)
    
    # Handle missing values
    # For this example, we'll fill numerical NaNs with the median and categorical NaNs with the mode
    for column in data.columns:
        if data[column].dtype in ['float64', 'int64']:
            data[column].fillna(data[column].median(), inplace=True)
        else:
            data[column].fillna(data[column].mode()[0], inplace=True)
    
    # Convert date columns to datetime format
    date_columns = ['clear_date', 'posting_date', 'document_create_date', 
                    'document_create_date.1', 'due_in_date', 'baseline_create_date']
    for date_column in date_columns:
        data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
    
    # Create a new feature: days_to_due from posting_date to due_in_date
    data['days_to_due'] = (data['due_in_date'] - data['posting_date']).dt.days
    
    return data

# Load and process the data
processed_data = load_and_process_data('OnlineSalesData.csv')

# Display the first few rows of the processed data
processed_data.head()