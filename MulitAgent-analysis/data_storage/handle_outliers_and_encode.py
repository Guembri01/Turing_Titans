# Handling outliers
# For simplicity, let's use the IQR method to cap outliers

def cap_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series.clip(lower_bound, upper_bound)

# Apply the capping function to numerical columns
data[numerical_cols] = data[numerical_cols].apply(cap_outliers)

# Encoding categorical variables
# Using one-hot encoding for categorical variables
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

data_encoded.head()