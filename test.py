import pandas as pd
from joblib import load
import numpy as np

# Function to calculate MAPE
def get_mape(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Load the model
best_model = load('best_model.joblib')

# Load the test data
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')  # This will return a DataFrame

# Convert y_test to a 1-D array, since your model will predict a 1-D array
y_test = y_test.values.flatten()

# Predict on test data
y_pred_test = best_model.predict(X_test)

# Calculate and print final MAPE
final_mape = get_mape(y_test, y_pred_test)
print(f"Final MAPE on test set: {final_mape} %")
