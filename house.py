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

model = LinearRegression(fit_intercept=True)
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

import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

# Tạo bins cho giá trị thực tế và giá trị dự đoán
bins = np.linspace(y_test.min(), y_test.max(), 20)  # Chia thành 20 bins
bin_centers = (bins[:-1] + bins[1:]) / 2  # Tính trung tâm của mỗi bin

# Gán mỗi giá trị của y_test và y_predict vào bin tương ứng
binned_y_test = np.digitize(y_test, bins)  # Gán y_test vào các bin
binned_y_predict = np.digitize(y_predict, bins)  # Gán y_predict vào các bin

# Tính trung bình giá trị thực tế và dự đoán trong mỗi bin
mean_y_test = [y_test[binned_y_test == i].mean() for i in range(1, len(bins))]
mean_y_predict = [y_predict[binned_y_predict == i].mean() for i in range(1, len(bins))]

# Vẽ biểu đồ so sánh giữa giá trị thực tế và giá trị dự đoán trong mỗi bin
plt.figure(figsize=(10, 6))
plt.plot(bin_centers, mean_y_test, label="Giá trị thực tế", color='blue', marker='o')
plt.plot(bin_centers, mean_y_predict, label="Giá trị dự đoán", color='red', marker='x')
plt.xlabel('Giá trị thực tế')
plt.ylabel('Giá trị dự đoán')
plt.title('Binned Correlation Analysis - So sánh Giá trị Thực tế và Dự đoán')
plt.legend()
plt.grid(True)
plt.show()