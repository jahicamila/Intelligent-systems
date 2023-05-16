import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense


# Load data from Excel file
data = pd.read_excel('Podaci_primjer.xlsx')

# Select rows with indices 400 to 600
data_train = data.iloc[398:599]

# Split data into input (X) and output (Y) variables
X_train = data_train.iloc[:, 4:].values
Y_train = data_train.iloc[:, 1].values
Y_train = Y_train.reshape(-1, 1)

# Scale input data to range [0, 1]
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Define RNN architecture
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Reshape((1, X_train.shape[1])),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Train and evaluate model on training data
results = []
for i in range(7):
    # Split data into training and testing sets
    X_train_final, X_test_final, Y_train_final, Y_test_final = train_test_split(X_train_scaled, Y_train, test_size=0.2)

    # Train model on training data
    model = create_model()
    history = model.fit(X_train_final, Y_train_final, epochs=100, validation_data=(X_test_final, Y_test_final), verbose=0)

    # Evaluate model on test data
    mse = model.evaluate(X_test_final, Y_test_final, verbose=0)
    results.append(mse)

# Plot expected results
X_all = data.iloc[:, 4:].values
Y_all = data.iloc[:, 1].values
#X_all = np.column_stack((X_all, np.zeros(len(X_all))))
X_all_scaled = scaler.transform(X_all)

plt.figure(figsize=(12, 6))
#plt.plot(Y_all, label='Original Data', color='blue')
colors = ['red', 'green', 'purple', 'orange', 'black', 'pink', 'gray']
for i, mse in enumerate(results):
    model = create_model()
    model.fit(X_train_scaled, Y_train, epochs=100, verbose=0)
    Y_pred = model.predict(X_all_scaled)
    plt.plot(range(398,599), Y_pred[398:599, 0], label=f'Model {i}', color=colors[i])
plt.xlabel('Sample Index')
plt.ylabel('Output 1')
plt.legend()
plt.show()
