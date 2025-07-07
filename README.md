
# Housing Price Prediction and Data Visualization

This project includes two Python scripts for analyzing housing price data:
1. `prediction.py`: Performs data normalization, trains a linear regression model, and evaluates its performance.
2. `correlation_heatmap.py`: Visualizes correlations between features using a heatmap.

---

## 1. prediction.py

### Overview
This script performs the following tasks:
- Loads a housing dataset from a CSV file.
- Normalizes the data using Min-Max Scaling.
- Splits the data into training and testing sets.
- Trains a Linear Regression model to predict housing prices.
- Evaluates the model using Mean Squared Error (MSE) and Mean Absolute Error (MAE).
- Visualizes the comparison between actual and predicted values using a Binned Correlation Analysis.

### Libraries Used
- `numpy`: Numerical operations
- `pandas`: Data manipulation
- `scikit-learn`: Machine learning and data preprocessing
- `matplotlib`: Data visualization

### Data Normalization
The data is normalized using `MinMaxScaler` from scikit-learn to scale features between 0 and 1.

### Model Training
A Linear Regression model is trained using the normalized data. The model's performance is evaluated using:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)

### Visualization
The script visualizes the difference between actual and predicted values using a Binned Correlation Analysis plot.

---

## 2. correlation_heatmap.py

### Overview
This script:
- Loads the housing dataset from a CSV file.
- Calculates the correlation matrix between features.
- Visualizes the correlation matrix using a heatmap.

### Libraries Used
- `pandas`: Data manipulation
- `seaborn`: Data visualization (heatmap)
- `matplotlib`: Plot rendering

### Correlation Heatmap
The heatmap shows the correlation between features, helping to identify:
- Positive correlations (closer to 1)
- Negative correlations (closer to -1)
- No correlation (around 0)

### Interpretation
- High positive correlation indicates that as one feature increases, the other tends to increase.
- High negative correlation indicates that as one feature increases, the other tends to decrease.

---

## Installation and Setup

1. **Clone the repository:**
    ```
    git clone https://github.com/khoaissleep/PREDICT_HOUSE_PRICE.git
    cd PREDICT_HOUSE_PRICE
    ```

2. **Create a virtual environment and activate it:**
    ```
    python3 -m venv venv
    source venv/bin/activate   # On Windows: venv\\Scripts\\activate
    ```

3. **Install the required libraries:**
    ```
    pip install -r requirements.txt
    ```

---

## Usage

1. **To run prediction.py:**
    ```
    python prediction.py
    ```

2. **To run correlation_heatmap.py:**
    ```
    python correlation_heatmap.py
    ```

Make sure the dataset CSV files are in the correct path as specified in the scripts.

---

## Notes
- Modify the file paths in the scripts as needed.
- Ensure that the dataset is preprocessed correctly to avoid errors during training or visualization.

---

## License
This project is open-source and available under the MIT License.

---

## Author
Created by `khoaissleep`
"""

# Lưu nội dung vào file README.md
with open('/mnt/data/README.md', 'w') as f:
    f.write(readme_content)

"/mnt/data/README.md"
