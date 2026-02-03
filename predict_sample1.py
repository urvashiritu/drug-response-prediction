import pickle
import numpy as np

# Load your trained model (choose one)
model_path = "models/xgboost_reg_model.pkl"  # You can change to another model if you want
model = pickle.load(open(model_path, "rb"))

# Example: create a fake input sample with 10 numerical features
# You can adjust the number of features if your dataset has more/less
sample_input = np.random.rand(1, 10)

# Make prediction
prediction = model.predict(sample_input)

print(f"Using model: {model_path}")
print("Sample input:", sample_input)
print("Predicted drug response:", prediction)
