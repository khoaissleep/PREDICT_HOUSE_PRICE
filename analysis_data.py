# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the dataset
readfile = pd.read_csv("/home/khoa_is_sleep/machine_learning/Housing_updated_processed.csv")

# Create a heatmap to visualize the correlation between features
sns.heatmap(readfile.corr(), annot=True, cmap='coolwarm', linewidths=0.5)

# Add titles and labels for better visualization
plt.title("Correlation Heatmap of Features")
plt.xlabel("Features")
plt.ylabel("Features")
plt.show()
