import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score, explained_variance_score, mean_absolute_percentage_error

# Define custom metrics
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Define the scoring metrics
scoring_metrics = {
    "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
    "MSE": make_scorer(mean_squared_error, greater_is_better=False),
    "RMSE": make_scorer(rmse, greater_is_better=False),
    "MAPE": make_scorer(mean_absolute_percentage_error, greater_is_better=False),
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
#final_df.drop('Injury Type', axis=1, inplace=True)

# Results dataframe
results = []

# Prepare the features (X) and target (y)
X = final_df.drop('Days Missed', axis=1)  # Features
y = final_df['Days Missed']  # Target variable

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

# Convert results to dataframe
cv_results = pd.DataFrame(results)

print(cv_results)

cv_results.to_csv("assets/models_scores.csv", index=False)