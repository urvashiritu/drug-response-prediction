import joblib
import pandas as pd

# Load the trained regression model
model_path = "models/xgboost_reg_model.pkl"
model = joblib.load(model_path)

# Load one sample from test features
sample_path = "data/processed/X_test.csv"
df = pd.read_csv(sample_path)

# Pick one random sample
sample_data = df.sample(1, random_state=42)

# Predict the drug response value
prediction = model.predict(sample_data)

print("\n=== Drug Response Prediction (Regression) ===")
print("Sample Input Features:")
print(sample_data.to_string(index=False))
print("\nPredicted Drug Response:", prediction[0])
