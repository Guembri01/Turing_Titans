import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create a classification report visualization
from sklearn.metrics import classification_report

# Sample data for classification report
report_data = {'Precision': [1.00, 1.00], 'Recall': [1.00, 1.00], 'F1-Score': [1.00, 1.00]}
index = ['Class 0', 'Class 1']
report_df = pd.DataFrame(report_data, index=index)

# Plotting the classification report
plt.figure(figsize=(10, 6))
sns.heatmap(report_df, annot=True, cmap='Blues', fmt='.2f')
plt.title('Classification Report Heatmap')
plt.xlabel('Metrics')
plt.ylabel('Classes')

# Save the figure
plt.savefig('classification_report_heatmap.png')
plt.close()