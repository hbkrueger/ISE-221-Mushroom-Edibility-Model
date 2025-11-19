import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, f1_score,
    recall_score, confusion_matrix
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# LOADING DATA -------------------------------------

train_df = pd.read_csv("train.csv")
val_df   = pd.read_csv("val.csv")
test_df  = pd.read_csv("test.csv")

target_col = "class_p"

y_train = train_df[target_col].astype(int)
y_val   = val_df[target_col].astype(int)
y_test  = test_df[target_col].astype(int)

X_train = train_df.drop(columns=[target_col])
X_val   = val_df.drop(columns=[target_col])
X_test  = test_df.drop(columns=[target_col])

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)

input_dim = X_train.shape[1]

# DEFINING MODELS -----------------------------------------------

# Model 1: ReLU, 32 -> 16 -> 1
model1 = Sequential([
    Dense(32, activation='relu', input_dim=input_dim),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
model1.compile(optimizer=Adam(learning_rate=0.001),
               loss='binary_crossentropy',
               metrics=['accuracy'])

# Model 2: all sigmoid, 32 -> 16 -> 1
model2 = Sequential([
    Dense(32, activation='sigmoid', input_dim=input_dim),
    Dense(16, activation='sigmoid'),
    Dense(1, activation='sigmoid')
])
model2.compile(optimizer=Adam(learning_rate=0.001),
               loss='binary_crossentropy',
               metrics=['accuracy'])

# Model 3: small ReLU, 5 -> 3 -> 1
model3 = Sequential([
    Dense(5, activation='relu', input_dim=input_dim),
    Dense(3, activation='relu'),
    Dense(1, activation='sigmoid')
])
model3.compile(optimizer=Adam(learning_rate=0.001),
               loss='binary_crossentropy',
               metrics=['accuracy'])

# TRAIN ALL MODELS WITH VALIDATION DATA ------------------------

history1 = model1.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    verbose=1
)

history2 = model2.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    verbose=1
)

history3 = model3.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    verbose=1
)

# LOSS CURVES -----------------------------

def plot_loss(history, title):
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

plot_loss(history1, "Model 1 Loss Curve (ReLU 32-16)")
plot_loss(history2, "Model 2 Loss Curve (Sigmoid 32-16)")
plot_loss(history3, "Model 3 Loss Curve (ReLU 5-3)")

# PREDICTIONS ON TEST SET ----------------------------------

y_true = y_test.astype(int)

y_pred1 = (model1.predict(X_test) > 0.5).astype(int)
y_pred2 = (model2.predict(X_test) > 0.5).astype(int)
y_pred3 = (model3.predict(X_test) > 0.5).astype(int)

# METRICS ---------------------------------------------------

def evaluate_model(y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    return mse, mae, acc, prec, rec, f1

metrics1 = evaluate_model(y_true, y_pred1)
metrics2 = evaluate_model(y_true, y_pred2)
metrics3 = evaluate_model(y_true, y_pred3)

results = pd.DataFrame({
    "Model": ["Model 1 (ReLU 32-16)", "Model 2 (Sigmoid 32-16)", "Model 3 (ReLU 5-3)"],
    "MSE":      [metrics1[0], metrics2[0], metrics3[0]],
    "MAE":      [metrics1[1], metrics2[1], metrics3[1]],
    "Accuracy": [metrics1[2], metrics2[2], metrics3[2]],
    "Recall":   [metrics1[4], metrics2[4], metrics3[4]],
    "F1 Score": [metrics1[5], metrics2[5], metrics3[5]],
})


print("\nModel Performance Comparison:\n")
print(results)

# CONFUSION MATRICES ---------------------------------

def plot_conf_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

plot_conf_matrix(y_true, y_pred1, "Model 1 Confusion Matrix")
plot_conf_matrix(y_true, y_pred2, "Model 2 Confusion Matrix")
plot_conf_matrix(y_true, y_pred3, "Model 3 Confusion Matrix")


# Debug-
# train_set = set(map(tuple, train_df.values))
# val_set   = set(map(tuple, val_df.values))
# test_set  = set(map(tuple, test_df.values))
#
# overlap_train_test_unique = train_set & test_set
# overlap_train_val_unique  = train_set & val_set
# overlap_val_test_unique   = val_set & test_set

# print("UNIQUE overlap TRAIN/TEST:", len(overlap_train_test_unique))
# print("UNIQUE overlap TRAIN/VAL:", len(overlap_train_val_unique))
# print("UNIQUE overlap VAL/TEST:", len(overlap_val_test_unique))
#
# print("Misclassified (Model 1):", (y_true.values != y_pred1.flatten()).sum())
# print("Misclassified (Model 2):", (y_true.values != y_pred2.flatten()).sum())
# print("Misclassified (Model 3):", (y_true.values != y_pred3.flatten()).sum())
#
# print("model2 == model3 pred: ", np.array_equal(y_pred2, y_pred3))
