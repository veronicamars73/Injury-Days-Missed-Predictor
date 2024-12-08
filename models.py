import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score, explained_variance_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt

 # Function to collect feature importances for each fold
def collect_importances(estimator, X, y):
    estimator.fit(X, y)
    # Store feature importances for the current fold
    if hasattr(estimator, 'feature_importances_'):
        return estimator.feature_importances_
    else:
        return np.zeros(X.shape[1]) 

# Define custom metrics
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred, epsilon=1e-5):
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

# Define the scoring metrics
scoring_metrics = {
    "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
    "MSE": make_scorer(mean_squared_error, greater_is_better=False),
    "RMSE": make_scorer(rmse, greater_is_better=False),
    "MAPE": make_scorer(mape, greater_is_better=False),
    "R²": make_scorer(r2_score),
    "Explained Variance": make_scorer(explained_variance_score),
}

# Define cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=7)


# Models to evaluate
models = {
    "Random Forest Regressor": RandomForestRegressor(random_state=7),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
}

# Read Preprocessed CSV
final_df = pd.read_csv('assets/final_injury_dataset_for_ml.csv')

# Results dataframe
results = []

# Prepare the features (X) and target (y)
X = final_df.drop('Days Missed', axis=1)  # Features
y = final_df['Days Missed']  # Target variable

# To store feature importances across all folds
feature_importance_dict = {col: [] for col in X.columns}

# Perform cross-validation
for model_name, model in models.items():
    for metric_name, scoring in scoring_metrics.items():
        scores = cross_val_score(model, X, y, cv=kf, scoring=scoring)
        mean_score = scores.mean()
        std_score = scores.std()
        results.append({
            "Model": model_name,
            "Metric": metric_name,
            "Mean Score": -mean_score if "MAE" in metric_name or "MSE" in metric_name or "RMSE" in metric_name or "MAPE" in metric_name else mean_score,
            "Std Dev": std_score,
        })
        
        fold_importances = np.array([collect_importances(model, X.iloc[train_index], y.iloc[train_index]) 
                                     for train_index, _ in kf.split(X)])
         # Average feature importances across all folds
        mean_importances = fold_importances.mean(axis=0)

        # Update the feature importance dictionary
        for feature, importance in zip(X.columns, mean_importances):
            feature_importance_dict[feature].append(importance)

# Convert results to dataframe
cv_results = pd.DataFrame(results)

print(cv_results)

cv_results.to_csv("assets/models_scores.csv", index=False)

# Convert the feature importance dictionary to a DataFrame
feature_importance_df = pd.DataFrame(feature_importance_dict)

# Plot the average feature importances (mean across all folds)
mean_importances = feature_importance_df.mean(axis=0)

# Sort the features by importance
sorted_importances = mean_importances.sort_values(ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.barh(sorted_importances.index, sorted_importances.values, color='skyblue')
plt.xlabel('Importance')
plt.title('Average Feature Importance Across Cross-Validation Folds')
plt.gca().invert_yaxis()  # To show the most important features at the top
plt.show()

print(sorted_importances)