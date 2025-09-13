import mlflow.sklearn
import pandas as pd

# Example test input (replace with real data later)
sample_data = {
    "Open": [50000],
    "High": [51000],
    "Low": [49000],
    "Close": [50500],
    "Volume": [1200000]
}
X_test = pd.DataFrame(sample_data)

# Load model from a specific MLflow run (use your run_id here)
model = mlflow.sklearn.load_model("9aa45b2f61a34c31920a1bc8d04a9214")

# Make prediction
preds = model.predict(X_test)
print("Predictions:", preds)
