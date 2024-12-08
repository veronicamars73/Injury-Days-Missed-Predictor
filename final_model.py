from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
    median_absolute_error
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read Preprocessed CSV
final_df = pd.read_csv('assets/final_injury_dataset_for_ml.csv')

# Prepare the features (X) and target (y)
X = final_df.drop(['Days Missed', 'Position'], axis=1)  # Features
y = final_df['Days Missed']  # Target variable

# Define the best hyperparameters based on previous evaluation
best_params = {
    'n_estimators': 500,
    'max_depth': 20,
    'min_samples_leaf': 2,
    'min_samples_split': 2,
    'bootstrap': True,
    'max_features': 'log2',
    'random_state': 7  # Fixed for reproducibility
}

# Initialize the RandomForestRegressor with the best hyperparameters
best_rf = RandomForestRegressor(**best_params)

# Train the model on the entire dataset (X, y)
best_rf.fit(X, y)

# Evaluate the model using cross-validation (5-fold)
cv_scores = cross_val_score(best_rf, X, y, cv=5, scoring='neg_mean_squared_error')

# Calculate the mean and standard deviation of the cross-validation MSE
mean_cv_mse = -cv_scores.mean()  # Convert from negative MSE
std_cv_mse = cv_scores.std()

# Predict on the training data for evaluation metrics (or use a hold-out test set if desired)
y_pred = best_rf.predict(X)

# Calculate evaluation metrics
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)
explained_variance = explained_variance_score(y, y_pred)
medae = median_absolute_error(y, y_pred)
mape = np.mean(np.abs((y - y_pred) / y)) * 100  # Mean Absolute Percentage Error

# Normalized RMSE and CVRMSE
nrmse = rmse / (y.max() - y.min())
cvrmse = rmse / y.mean()

# Residual analysis
residuals = y - y_pred
residual_mean = np.mean(residuals)
residual_std = np.std(residuals)

# Feature importance
feature_importances = pd.DataFrame(
    {
        'Feature': X.columns,
        'Importance': best_rf.feature_importances_
    }
).sort_values(by='Importance', ascending=False)

# Display evaluation results
print("Cross-Validation Mean MSE: ", mean_cv_mse)
print("Cross-Validation Std MSE: ", std_cv_mse)
print("Training MAE: ", mae)
print("Training MSE: ", mse)
print("Training RMSE: ", rmse)
print("Training R²: ", r2)
print("Training Explained Variance: ", explained_variance)
print("Training MedAE: ", medae)
print("Training MAPE: ", mape, "%")
print("Normalized RMSE: ", nrmse)
print("CVRMSE: ", cvrmse)
print("Residual Mean: ", residual_mean)
print("Residual Std: ", residual_std)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.show()

# Residual plot
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()
