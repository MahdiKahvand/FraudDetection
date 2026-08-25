import json
import os

import joblib
import numpy as np

# ============================================================
# 1. Configuration
# ============================================================

MODEL_DIR = "models"
MODEL_NAME = "bestmodel.pkl"
SCALER_NAME = "scaler.pkl"
INPUT_NAME = "input.json"
OUTPUT_NAME = "output.json"

# ============================================================
# 2. Create Paths
# ============================================================

model_path = os.path.join(
    "..",
    MODEL_DIR,
    MODEL_NAME
)

scaler_path = os.path.join(
    "..",
    MODEL_DIR,
    SCALER_NAME
)

input_path = os.path.join(
    "..",
    INPUT_NAME
)

output_path = os.path.join(
    "..",
    OUTPUT_NAME
)


# ============================================================
# 3. Load Model and Scaler
# ============================================================

print("========== Loading model and scaler ==========")
print()

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

print("Model and scaler loaded successfully.")
print()


# ============================================================
# 4. Load Input Data
# ============================================================

print("========== Loading input data ==========")
print()

with open(input_path, "r") as file:
    data = json.load(file)

print("Input data loaded successfully.")
print()


# ============================================================
# 5. Prepare Input
# ============================================================

input_sample = np.array(
    list(data.values())
).reshape(1, -1)

input_scaled = scaler.transform(input_sample)


# ============================================================
# 6. Make Prediction
# ============================================================

predict = model.predict(input_scaled)

probability = model.predict_proba(input_scaled)


# ============================================================
# 7. Process Prediction Result
# ============================================================

# Get predicted class ID
class_id = int(predict[0])

# Get probability of predicted class
class_probability = probability[0, class_id]

# Class mapping
classing_dict = {
    0: "Normal",
    1: "Fraud"
}

class_name = classing_dict[class_id]


# ============================================================
# 8. Display Prediction Result
# ============================================================




print("========== Prediction Result ==========")
print()

print(f"Class          : {class_name}")
print(f"Class ID       : {class_id}")
print(f"Probability    : {float(class_probability):.4f}")
print(f"Probability (%) : {float(class_probability):.2%}")

result_tabel = {
    "Class" : class_name ,
    "Class ID" : class_id ,
    "Probability" : float(class_probability) ,
    "Probability (%)" : float(class_probability * 100)
}


with open(output_path, "w") as file:
    json.dump(result_tabel, file, indent=4)