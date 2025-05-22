from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model
MODEL_PATH = "model/model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Run train.py first.")

model = joblib.load(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Check for 'features'
        if "features" not in data:
            return jsonify({"error": "Missing 'features' key in JSON"}), 400

        features = data["features"]

        if not isinstance(features, list) or len(features) != 30:
            return jsonify({"error": "Expected 'features' to be a list of 30 numeric values"}), 400

        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]
        result = "fraud" if prediction == 1 else "not fraud"

        return jsonify({"prediction": result})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def index():
    return "Credit Card Fraud Prediction API is running."

if __name__ == "__main__":
    app.run(debug=True, port=5001)
