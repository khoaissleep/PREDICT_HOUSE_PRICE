# Import necessary libraries
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
import matplotlib.pyplot as plt

# Ignore warnings from sklearn
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# File path to the dataset
file_path = "/home/khoa_is_sleep/machine_learning/Housing_filtered_columns_with_price.csv"
readfile = pd.read_csv(file_path)

# Display initial data
print("Initial Data:")
print(readfile.head())

# # Z-score normalization
# st = StandardScaler()
# readfile_normal = pd.DataFrame(
#     st.fit_transform(readfile), columns=readfile.columns
# )

# Linear scaling normalization
scaler = MinMaxScaler()
readfile_normal = pd.DataFrame(scaler.fit_transform(readfile), columns=readfile.columns)

print("\nData after linear scaling normalization:")
print(readfile_normal.head())

# Define features and target
X = readfile_normal[["area", "bathrooms", "bedrooms", "stories", "airconditioning", "parking"]]
y = readfile_normal["price"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)

# Make predictions
y_predict = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_predict)
mae = mean_absolute_error(y_test, y_predict)

# Display model results
print("\nModel Results:")
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Root Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)

# Binned Correlation Analysis for visualization
# Create bins for actual and predicted values
bins = np.linspace(y_test.min(), y_test.max(), 20)  # Divide into 20 bins
bin_centers = (bins[:-1] + bins[1:]) / 2  # Calculate the center of each bin

# Assign each value of y_test and y_predict to the corresponding bin
binned_y_test = np.digitize(y_test, bins)  # Bin y_test
binned_y_predict = np.digitize(y_predict, bins)  # Bin y_predict

# Calculate the mean of actual and predicted values in each bin
mean_y_test = [y_test[binned_y_test == i].mean() for i in range(1, len(bins))]
mean_y_predict = [y_predict[binned_y_predict == i].mean() for i in range(1, len(bins))]

# Plot comparison between actual and predicted values in each bin
plt.figure(figsize=(10, 6))
plt.plot(bin_centers, mean_y_test, label="Actual Values", color='blue', marker='o')
plt.plot(bin_centers, mean_y_predict, label="Predicted Values", color='red', marker='x')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Binned Correlation Analysis - Actual vs Predicted')
plt.legend()
plt.grid(True)
plt.show()
