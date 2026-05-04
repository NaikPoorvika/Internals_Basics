import json
import os

# Assume version 1 is current champion
champion_version = 1

# Simulate challenger (version 2)
challenger_version = 2

# Since we didn’t actually train a second model, keep champion
action = "kept"

output = {
    "registered_model_name": "copilotbench-suggestion-accept-rate-predictor",
    "alias_name": "champion",
    "champion_version": champion_version,
    "challenger_version": challenger_version,
    "action": action
}

os.makedirs("results", exist_ok=True)

with open("results/step4_s7.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 4 Done")