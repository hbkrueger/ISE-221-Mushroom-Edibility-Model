import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

# X_train, y_train as numpy arrays
df = pd.read_csv("train.csv")

y_train = df["class_p"] # class_p
X_train = df.drop('class_p', axis=1) # every col except class p, 20 cols for all features

input_dim = X_train.shape[1] # number of cols = 20

model1 = Sequential([
    Dense(32, activation='relu', input_dim=input_dim), # hidden layer 1, 32 nodes w/ relu activation func.
    Dense(16, activation='relu'), # hidden layer 2, 16 nodes w/ relu activation func.
    Dense(1, activation='sigmoid')  # output layer, w/ sigmoid activation func. will output 0 -> 1
])

# try to minimize binary cross-entropy
model1.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# fit model with data, 20 epochs, checks 32 rows before updating weights 
history1 = model1.fit(X_train, 
                    y_train, 
                    epochs=10, 
                    batch_size=32)

# =====================================================================
model2 = Sequential([
    Dense(32, activation='sigmoid', input_dim=input_dim), # hidden layer 1, 32 nodes w/ sigmoid activation func.
    Dense(16, activation='sigmoid'), # hidden layer 2, 16 nodes w/ sigmoid activation func.
    Dense(1, activation='sigmoid')  # output layer, w/ sigmoid activation func. will output 0 -> 1
])

# try to minimize binary cross-entropy
model2.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# fit model with data, 20 epochs, checks 32 rows before updating weights 
history2 = model1.fit(X_train, 
                    y_train, 
                    epochs=10, 
                    batch_size=32)
# ==============================================================================================
model1 = Sequential([
    Dense(32, activation='relu', input_dim=input_dim), # hidden layer 1, 32 nodes w/ relu activation func.
    Dense(16, activation='relu'), # hidden layer 2, 16 nodes w/ relu activation func.
    Dense(1, activation='sigmoid')  # output layer, w/ sigmoid activation func. will output 0 -> 1
])

# try to minimize binary cross-entropy
model1.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# fit model with data, 20 epochs, checks 32 rows before updating weights 
history1 = model1.fit(X_train, 
                    y_train, 
                    epochs=10, 
                    batch_size=32)

# =====================================================================
model3 = Sequential([
    Dense(5, activation='relu', input_dim=input_dim), # hidden layer 1, 5 nodes w/ relu activation func.
    Dense(3, activation='relu'), # hidden layer 2, 3 nodes w/ relu activation func.
    Dense(1, activation='sigmoid')  # output layer, w/ sigmoid sigmoid func. will output 0 -> 1
])

# try to minimize binary cross-entropy
model3.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# fit model with data, 20 epochs, checks 32 rows before updating weights 
history3 = model3.fit(X_train, 
                    y_train, 
                    epochs=10, 
                    batch_size=32)
# ====================================
# TODO add testing data here to see when to stop training
#plt.plot(history1.history["loss"], c="r")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(history1.history["loss"], c="r")
plt.legend(["model1"])
plt.show()
