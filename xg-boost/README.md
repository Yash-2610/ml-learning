# XGBoost Regression  Insurance Charges Prediction

This project uses `XGBRegressor` to predict medical insurance charges based on customer details.  
The model is tuned using `GridSearchCV` and can also take user input to make a custom prediction.

---

## Dataset

The dataset `insurance.csv` contains:

- **Numerical features**:
  - `age`
  - `bmi`
  - `children`
- **Categorical features**:
  - `sex`
  - `smoker`
  - `region`
- **Target**:
  - `charges` (insurance cost)



## Preprocessing
- The target y is set as charges.

- All other columns are used as features x.

- Categorical features are converted to numeric with one-hot encoding:

``` x = pd.get_dummies(df.drop(columns=['charges']), drop_first=True).astype(float)```


# Base model:
```model = XGBRegressor(n_estimators=300, max_depth=6)```
Hyperparameter grid:

Model and Hyperparameter Tuning

`n_estimators: [40, 60, 80]`

`learning_rate: [0.1, 0.15]`

`max_depth: [2, 3, 4]`

`gamma: [2, 2.22]`

`subsample: [0.7, 0.8]`

`colsample_bytree: [0.7, 0.8]`

# Grid search:
```grid = GridSearchCV(model, param_grid=parameters, cv=4)grid.fit(x_train, y_train)```

# Evaluation:

`y_pred = grid.predict(x_test)` 

`mean_squared_error(y_test, y_pred)`

`r2_score(y_test, y_pred)`

The best hyperparameters from the search are printed along with MSE and R².

# User Input Prediction:

After training, the script allows the user to enter their own data:

Numeric inputs: age, bmi, children
Categorical inputs: sex, smoker, region

The steps are:
- Collect input into a dictionary.
- Convert to a DataFrame:


`user_df = pd.DataFrame([user_data])`
- One-hot encode and align with training columns:

`user_encoded = pd.get_dummies(user_df, drop_first=True).astype(float)`

- Reindex with original data

`user_encoded = user_encoded.reindex(columns=x.columns, fill_value=0)`

- Predict using the best model from grid search:

`user_pred = grid.best_estimator_.predict(user_encoded)`

The final output is the predicted insurance charge for the entered user details.

# Summary
Uses XGBoost for regression on insurance charges.

Handles categorical features via one-hot encoding.

Tunes hyperparameters with GridSearchCV.

Evaluates performance with MSE and R².

Includes an interactive prediction step for custom user inputs.