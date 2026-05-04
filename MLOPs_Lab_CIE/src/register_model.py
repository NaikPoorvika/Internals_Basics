import mlflow

run_id = "a0fd8a97f0274d20b18291f9745fbf13"

model_uri = f"runs:/{run_id}/RandomForest"

mlflow.register_model(
    model_uri,
    "copilotbench-suggestion-accept-rate-predictor"
)

print("Model registered")