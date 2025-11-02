
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import pandas as pd

# Load and preprocess the data
data = pd.read_csv("mushrooms.csv", na_values="?")
data_clean = data.dropna()

# Separate features and target
X = data_clean.drop('class', axis=1)
y = data_clean['class']

# change all letters to numbers
le = LabelEncoder()
X_encoded = X.apply(lambda col: le.fit_transform(col))
y_encoded = le.fit_transform(y)

# Compute mutual information, how much each column reduces uncertinty
mi = mutual_info_classif(X_encoded, y_encoded, discrete_features=True)

#visualize the graph
mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)

# Plot using a lolipop graph
plt.figure(figsize=(12,6))
plt.hlines(y=mi_series.index, xmin=0, xmax=mi_series.values, color='skyblue')
plt.plot(mi_series.values, mi_series.index, "o")
plt.title('How Much Each Feature Predicts Mushroom Edibility')
plt.xlabel('Reduced uncertainty')
plt.ylabel('Features')
plt.show()