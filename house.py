import numpy as np
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


file_path = "/home/khoa_is_sleep/machine_learning/Housing_filtered_columns_with_price.csv"
readfile = pd.read_csv(file_path)

print("Dữ liệu ban đầu:")
print(readfile.head())

    # # chuẩn hóa dữ liệu theo thang điểm z
    # st = StandardScaler()
    # readfile_normal = pd.DataFrame(
    #     st.fit_transform(readfile), columns=readfile.columns
    # )
# linear scaling
scaler = MinMaxScaler()
readfile_normal= pd.DataFrame(scaler.fit_transform(readfile), columns=readfile.columns)

print("\nDữ liệu sau khi chuẩn hóa theo linear scaling:")
print(readfile_normal.head())

x = readfile_normal[["area", "bathrooms", "bedrooms", "stories", "airconditioning", "parking"]]
y = readfile_normal["price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

mse = mean_squared_error(y_test, y_predict)
mae = mean_absolute_error(y_test, y_predict)

print("\nKết quả mô hình:")
print("Coefficients (Hệ số):", model.coef_)
print("Intercept (Giao điểm):", model.intercept_)
print("Root Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)

import matplotlib.pyplot as plt

# Plot the true vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_predict, color='blue', label='Predicted vs True Values')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal Fit Line')

plt.title("Linear Regression: Predicted vs True Prices")
plt.xlabel("True Prices")
plt.ylabel("Predicted Prices")
plt.legend()
plt.grid(True)
plt.show()
