import pandas as pd
import seaborn as ss
import matplotlib.pyplot as mat
from sklearn.preprocessing import StandardScaler

readfile= pd.read_csv("/home/khoa_is_sleep/machine_learning/Housing_updated_processed.csv")




ss.heatmap(readfile.corr(), annot=True ,cmap='coolwarm',linewidths=0.5)

mat.title("visualize data")
mat.xlabel("features")
mat.ylabel("features")
mat.show()
