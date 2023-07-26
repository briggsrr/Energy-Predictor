import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from joblib import dump


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
set_seed(420)


df = pd.read_csv('CCPP_full_data.csv')

# EXPLORATION
print(df.describe())
print(df.head())
print(df.info())
plt.figure(figsize = (7, 5))
sns.heatmap(df.corr(), annot = True)
plt.show()

#setting up data 
X = df.drop(['PE', 'V'], axis=1)
y = df['PE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=420)

#saving data
X_test.to_csv('X_test.csv', index=False)
y_test.to_csv('y_test.csv', index=False)


cv = KFold(n_splits=5, random_state=420, shuffle=True)

# Training model1 with n_estimators=100 and max_depth=5
rf1 = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=420)
cv_scores1 = cross_val_score(rf1, X_train, y_train, cv=cv, scoring='neg_mean_absolute_error')
mae1 = -cv_scores1.mean()  # cv_scores will be negative because of the 'neg_mean_absolute_error' scoring

# Training model2 with n_estimators=200 and max_depth=10
rf2 = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=420)
cv_scores2 = cross_val_score(rf2, X_train, y_train, cv=cv, scoring='neg_mean_absolute_error')
mae2 = -cv_scores2.mean()

# Training model3 with n_estimators=300 and max_depth=15
rf3 = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=420)
cv_scores3 = cross_val_score(rf3, X_train, y_train, cv=cv, scoring='neg_mean_absolute_error')
mae3 = -cv_scores3.mean()

print(f"Cross-validated MAE for model 1: {mae1}")
print(f"Cross-validated MAE for model 2: {mae2}")
print(f"Cross-validated MAE for model 3: {mae3}")

best_mape = min([mae1, mae2, mae3])
if best_mape == mae1:
    best_model = rf1
elif best_mape == mae2:
    best_model = rf2
else:
    best_model = rf3


best_model.fit(X_train, y_train)
dump(best_model, 'best_model.joblib') 
