import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
# Assuming the dataset is in the same directory and named 'OnlineSalesData.csv'
df = pd.read_csv('OnlineSalesData.csv')

# Create a bar plot for the accuracy of the model
accuracy = [0.9999]
labels = ['Model Accuracy']

plt.figure(figsize=(8, 5))
sns.barplot(x=labels, y=accuracy, palette='viridis')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

# Save the figure
plt.savefig('model_accuracy.png')
plt.close()