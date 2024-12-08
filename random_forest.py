from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
import numpy as np
import pandas as pd

# Read Preprocessed CSV
final_df = pd.read_csv('assets/final_injury_dataset_for_ml.csv')

# Prepare the features (X) and target (y)
X = final_df.drop(['Days Missed','Position'], axis=1)  # Features (all columns except 'Days Missed' and position)
y = final_df['Days Missed']  # Target variable

# Initialize lists to store metrics across random states
random_states = range(15)  # Use 5 different random states
metrics_results = []

# Loop through different random states
for random_state in random_states:
    print(f"Evaluating with random_state={random_state}...")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    # Define the Random Forest model
    rf = RandomForestRegressor()
    
    # Define the hyperparameter grid
    param_grid = {
    'n_estimators': [500, 525, 550, 600],
    'min_samples_split': [2, 3, 5, 7],
    'min_samples_leaf': [1, 2, 3, 4],
    'max_features': ['log2'],  # Fixing based on analysis
    'max_depth': [20, 30, 40, None],
    'bootstrap': [True, False]
}
    
    # Set up RandomizedSearchCV with 5-fold cross-validation
    random_search = RandomizedSearchCV(
        estimator=rf, param_distributions=param_grid, 
        n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1
    )
    
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
    
    # Store metrics for this random state
    metrics_results.append({
        "Random State": random_state,
        "Best Parameters": random_search.best_params_,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R²": r2,
        "Explained Variance": explained_variance
    })

# Create a DataFrame for results
results_df = pd.DataFrame(metrics_results)

# Calculate mean and std deviation of metrics across random states
summary = results_df.describe().T[['mean', 'std']]

# Print the summary and results
print("\nResults Summary Across Random States:")
print(summary)

print("\nDetailed Results for Each Random State:")
print(results_df)

results_df.to_csv("assets/parameters.csv", index=False)