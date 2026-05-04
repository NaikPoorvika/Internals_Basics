import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import os
import joblib

# Load data
df = pd.read_csv("data/training_data.csv")

X = df.drop("suggestion_accept_rate", axis=1)
y = df["suggestion_accept_rate"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("copilotbench-suggestion-accept-rate")

results = []

def evaluate(model, name):
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        mlflow.log_param("model", name)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.set_tag("team", "ml_engineering")

        mlflow.sklearn.log_model(model, name)

        return {
            "name": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        }, model

# Train models
lr_res, lr_model = evaluate(LinearRegression(), "LinearRegression")
rf_res, rf_model = evaluate(RandomForestRegressor(random_state=42), "RandomForest")

results.extend([lr_res, rf_res])

# Select best model
best = min(results, key=lambda x: x["mae"])

# Save best model
os.makedirs("models", exist_ok=True)
joblib.dump(
    rf_model if best["name"] == "RandomForest" else lr_model,
    "models/best_model.pkl"
)

# Save JSON
output = {
    "experiment_name": "copilotbench-suggestion-accept-rate",
    "models": results,
    "best_model": best["name"],
    "best_metric_name": "mae",
    "best_metric_value": best["mae"]
}

os.makedirs("results", exist_ok=True)
with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 1 Done")