import pandas as pd
from joblib import load
import numpy as np

# Function to calculate MAPE
def get_mae(y_true, y_pred): 
     return np.mean(np.abs(y_true - y_pred))

best_model = load('best_model.joblib')

# Load the test data
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')
y_test = y_test.iloc[:, 0] 
# Make predictions with the best model
y_pred_test = best_model.predict(X_test)
final_mae = get_mae(y_test, y_pred_test)

print(f"Final MAE on test set: {final_mae} ")
