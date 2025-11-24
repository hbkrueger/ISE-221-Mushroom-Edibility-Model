import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam #learning rate
from tensorflow.keras.regularizers import l2 #reguralization
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

train_df = pd.read_csv("train.csv")
val_df   = pd.read_csv("val.csv")

X_train = train_df.drop("class_p", axis=1)
y_train = train_df["class_p"]

X_val = val_df.drop("class_p", axis=1)
y_val = val_df["class_p"]

input_dim = X_train.shape[1]

# Hyperparameter grids
learning_rates = [0.01, 0.001, 0.0001]
lambdas = [0, 0.001, 0.01, 0.1]

results = []
#==================model1==================#
for lr in learning_rates: #create 12 model1s with different reguralization and learning rates using a double loop
    for reg in lambdas:
        model1 = Sequential([ #add kernal_regularizer
            Dense(32, activation='relu', input_dim=input_dim,kernel_regularizer=l2(reg)), # hidden layer 1, 32 nodes w/ relu activation func.
            Dense(16, activation='relu',kernel_regularizer=l2(reg)), # hidden layer 2, 16 nodes w/ relu activation func.
            Dense(1, activation='sigmoid')  # output layer, w/ sigmoid activation func. will output 0 -> 1
        ])

        # try to minimize binary cross-entropy
        model1.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])

        # fit model with data, 20 epochs, checks 32 rows before updating weights
        history1 = model1.fit(X_train, y_train, epochs=10, batch_size=32, verbose = 0)

        val_pred = (model1.predict(X_val) > 0.5).astype(int)
        acc = accuracy_score(y_val, val_pred)

        results.append((lr, reg, acc))
#==================Result printing===================#

results_df = pd.DataFrame(results, columns=["Learning Rate", "L2 Reg", "Validation Accuracy"]) #put restults in a table to add to our report
print(results_df)

#heatmap printout
pivot = results_df.pivot(index="Learning Rate", columns="L2 Reg", values="Validation Accuracy")
plt.figure(figsize=(8,6))
sns.heatmap(pivot, annot=True, cmap="Blues")
plt.title("Grid Search Validation Accuracy")
plt.show()