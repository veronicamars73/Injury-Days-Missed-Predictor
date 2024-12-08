from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
import numpy as np
import pandas as pd

# Read Preprocessed CSV
final_df = pd.read_csv('assets/final_injury_dataset_for_ml.csv')

# Prepare the features (X) and target (y)
X = final_df.drop(['Days Missed'], axis=1)  # Features (all columns except 'Days Missed')
y = final_df['Days Missed']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,  random_state=7)

# Define the Random Forest model
rf = RandomForestRegressor()

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [400, 500, 600, 700, 1000],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [1, 2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 6],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False],
}

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, 
                                   n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1)

# Fit the model on training data
random_search.fit(X_train, y_train)

# Get the best model
best_rf = random_search.best_estimator_

# Evaluate the model on the test set
y_pred = best_rf.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
explained_variance = explained_variance_score(y_test, y_pred)

# Print the results
print("Best Parameters:", random_search.best_params_)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)
print("Explained Variance:", explained_variance)
