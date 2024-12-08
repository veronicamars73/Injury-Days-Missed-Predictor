from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Read Preprocessed CSV
final_df = pd.read_csv('assets/final_injury_dataset_for_ml.csv')

# Prepare the features (X) and target (y)
X = final_df.drop(['Days Missed', 'Position'], axis=1)  # Features (all columns except 'Days Missed' and 'Position')
y = final_df['Days Missed']  # Target variable

# Initialize lists to store metrics across random states
random_states = range(5, 15)  # Use 5 random states for performance reasons
metrics_results = []

# Loop through different random states
for random_state in random_states:
    print(f"Evaluating with random_state={random_state}...")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    # Define the Random Forest model
    rf = RandomForestRegressor(random_state=random_state)
    
    # Define the hyperparameter grid
    param_grid = {
        'n_estimators': [500, 550],
        'max_depth': [20, 25, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False],
        'max_features': ['log2']
    }
    
    # Set up GridSearchCV with 5-fold cross-validation
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        verbose=2,
        n_jobs=-1,
        scoring='neg_mean_squared_error'  # Scoring metric for evaluation
    )
    
    # Fit the grid search on training data
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_rf = grid_search.best_estimator_
    
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
        "Best Parameters": grid_search.best_params_,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R²": r2,
        "Explained Variance": explained_variance
    })

    # Feature importance plot for the best model
    if random_state == random_states[-1]:  # Generate plot for the last random state as an example
        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': best_rf.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importances)
        plt.title(f'Feature Importance - Best Model (Random State {random_state})')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.show()

# Create a DataFrame for results
results_df = pd.DataFrame(metrics_results)

# Calculate mean and standard deviation of metrics across random states
summary = results_df.describe().T[['mean', 'std']]

# Print the summary and results
print("\nResults Summary Across Random States:")
print(summary)

print("\nDetailed Results for Each Random State:")
print(results_df)

# Save results to CSV
results_df.to_csv("assets/parameters.csv", index=False)
