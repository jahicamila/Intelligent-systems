import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Load data from Excel file
data = pd.read_excel('Podaci_primjer.xlsx')

# Select rows with indices 400 to 600
data_train = data.iloc[398:599]

# Split data into input (X) and output (Y) variables
X_train = data_train.iloc[:, 4:].values
Y_train = data_train.iloc[:, 1].values
Y_train = Y_train.reshape(-1, 1)

# Scale input data 
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Split data into training and testing sets
X_train_final, X_test_final, Y_train_final, Y_test_final = train_test_split(X_train_scaled, Y_train, test_size=0.2)

# Define RNN architecture
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_final.shape[1],)),
    tf.keras.layers.Reshape((1, X_train_final.shape[1])),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(256, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile model with mean squared error loss and Adam optimizer
model.compile(loss='mean_squared_error', optimizer='adam')

# Train model on training data
history = model.fit(X_train_final, Y_train_final, epochs=100, validation_data=(X_test_final, Y_test_final))

# Evaluate model on test data
mse = model.evaluate(X_test_final, Y_test_final)
print(f"Test MSE: {mse:.4f}")

# Plot original dataset and verification data
X_all = data.iloc[:, 4:].values
Y_all = data.iloc[:, 1].values
X_all_scaled = scaler.transform(X_all)
Y_pred = model.predict(X_all_scaled)

plt.figure(figsize=(12, 6))
plt.plot(Y_all, label='Original Data', color='blue')
plt.plot(range(398, 599), Y_pred[398:599, 0], label='Verification Data', color='orange')
plt.xlabel('Sample Index')
plt.ylabel('Output 1')
plt.legend()
plt.show()
