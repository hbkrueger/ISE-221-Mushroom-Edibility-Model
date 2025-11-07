import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('cleaned_normalized_mushroom.csv')

# Separate features and target
X = df.drop('class_p', axis=1)
y = df['class_p']  # make the first column the y target

# First split: 70% train, 30% validation and testing
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Second split: split the 30% from above into 15% val, 15% test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Make the 3 different CSV files for the 3 different data sets
train = X_train.copy()
train['class_p'] = y_train

val = X_val.copy()
val['class_p'] = y_val

test = X_test.copy()
test['class_p'] = y_test

# Save them as single CSV files
train.to_csv('train.csv', index=False)
val.to_csv('val.csv', index=False)
test.to_csv('test.csv', index=False)

