import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, explained_variance_score

# Função para calcular as métricas de avaliação
def evaluate_model(y_test, predictions, model_name):
    return {
        "Model": model_name,
        "MAE": mean_absolute_error(y_test, predictions),
        "MSE": mean_squared_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "MAPE": mean_absolute_percentage_error(y_test, predictions),
        "R²": r2_score(y_test, predictions),
        "Explained Variance": explained_variance_score(y_test, predictions)
    }

# Read Preprocessed CSV
final_df = pd.read_csv('assets/final_injury_dataset_for_ml.csv')
#final_df.drop('Injury Type', axis=1, inplace=True)

# Prepare the features (X) and target (y)
X = final_df.drop('Days Missed', axis=1)  # Features (all columns except 'Days Missed')
y = final_df['Days Missed']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Avaliação dos modelos
models_results = []

""" Baseline models """

# Ridge Regression model (used as baseline model)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
models_results.append(evaluate_model(y_test, ridge_pred, "Ridge Regression"))

#  Lasso Regression model (baseline model)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
models_results.append(evaluate_model(y_test, lasso_pred, "Lasso Regression"))

""" Random Forest Regressor """
# Predict and evaluate the Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
models_results.append(evaluate_model(y_test, rf_pred, "Random Forest Regressor"))

# Criar o dataframe com as métricas
metrics_df = pd.DataFrame(models_results)
metrics_df = metrics_df.sort_values(by="R²", ascending=False)
print(metrics_df)