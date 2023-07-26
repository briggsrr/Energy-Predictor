# Energy-Predictor
For predicting the electrical energy output of a CCPP

# Use (first time)
`conda env create -f environment.yml`
`conda activate ner_system`
`python test.py`


### Modeling approach 
Ths is a supervised regression task. 

Could use MAE, MAPE, or MSE.
I am chosing MAE since it is easy to interpret in the context of the problem and is robust to outliers.

Could use linear, ridge, lasso, decision-tree, or random forest regression. 
I am chosing random forest regressor to handle multiple feature, robustness to outliers, and the lack of need for regularization. 

Could use tempurature (T), Ambient Pressure (AP), Relative Humdity (RH) or Exhaust Vacuum(V) for features. Based on correlation matrix, I will use AT, AP, and RH as features to avoid multicollinearity.


### Model building  
I trained 3 different models with varied hyperparameters (n_estimators and max_depth) using the sklearn package with 5-fold cross validation

## Model evaluation
I used the MAE metric, and it is produced in train.py

## Model interpretation - did you correctly interpret and clearly communicate the performance of your model?
I used Mean Absolute Error, which is robust to outliers, influenced by scale, and easy to interpret. On the test set, my best model got a MAE of ~3.1  meaning that on average, my model's predictions are off by 3.1 megawatts from the actual values, which is pretty good.