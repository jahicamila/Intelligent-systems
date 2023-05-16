import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# Load data from Excel file
data = pd.read_excel('Podaci_primjer.xlsx')                     

# Split data into input (X) and output (Y) variables
X = data.iloc[0:501, [4, 5, 6, 7, 8]].values
y = data.iloc[0:501, 1].values

# Scale input data
scaler = MinMaxScaler()
data[data.columns] = scaler.fit_transform(data[data.columns])

# Define the class limits and assign the classes
class_limit = 299
y = [1 if i < class_limit else 2 for i in y]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Train the MLP classification with different optimizers
mlp_adam = MLPClassifier(hidden_layer_sizes=(100, 200), max_iter=3000, solver='adam', random_state=100, activation='tanh')
mlp_sgd = MLPClassifier(hidden_layer_sizes=(100, 200), max_iter=3000, solver='sgd', learning_rate_init=0.01, random_state=100, activation='tanh')
mlp_lbfgs = MLPClassifier(hidden_layer_sizes=(100, 200), max_iter=3000, solver='lbfgs', random_state=100, activation='tanh')

mlp_adam.fit(X_train, y_train)
mlp_sgd.fit(X_train, y_train)
mlp_lbfgs.fit(X_train, y_train)

# Evaluate the MLP classification 
y_pred_adam = mlp_adam.predict(X_test)
y_pred_sgd = mlp_sgd.predict(X_test)
y_pred_lbfgs = mlp_lbfgs.predict(X_test)

print("Classification report (adam):")
print(classification_report(y_test, y_pred_adam))
print("Confusion matrix (adam):")
print(confusion_matrix(y_test, y_pred_adam))

print("Classification report (sgd):")
print(classification_report(y_test, y_pred_sgd))
print("Confusion matrix (sgd):")
print(confusion_matrix(y_test, y_pred_sgd))

print("Classification report (lbfgs):")
print(classification_report(y_test, y_pred_lbfgs))
print("Confusion matrix (lbfgs):")
print(confusion_matrix(y_test, y_pred_lbfgs))
