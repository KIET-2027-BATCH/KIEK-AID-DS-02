from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow CORS for all routes

# Load trained model and scaler
model_path = "models/electricity_model.pkl"  # Update path
scaler_path = "models/scaler.pkl"  # Update path



try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("✅ Model and Scaler loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model or scaler: {e}")
    model, scaler = None, None  # Prevent further errors

# Serve frontend HTML page
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model or scaler not loaded"}), 500

    try:
        data = request.get_json()
        print("🔹 Received JSON Data:", data)  # Debugging print

        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        required_features = ["production", "nuclear", "wind", "hydroelectric", "oilGas", "coal", "solar", "biomass"]
        missing_features = [feature for feature in required_features if feature not in data]

        if missing_features:
            return jsonify({"error": f"Missing required features: {', '.join(missing_features)}"}), 400

        # Convert input data to numpy array with validation
        try:
            features = np.array([float(data[feature]) for feature in required_features], dtype=np.float64).reshape(1, -1)
            print("🔹 Features before scaling:", features)  # Debugging print
        except ValueError as e:
            return jsonify({"error": f"Invalid input: {e}"}), 400

        # Scale input data
        try:
            features_scaled = scaler.transform(features)
            print("🔹 Features after scaling:", features_scaled)  # Debugging print
        except Exception as e:
            return jsonify({"error": f"Scaling error: {e}"}), 500

        # Make prediction
        try:
            prediction = model.predict(features_scaled)[0]
            print("🔹 Model Prediction:", prediction)  # Debugging print
            return jsonify({"prediction": float(prediction)})  # Convert to float for JSON serialization
        except Exception as e:
            return jsonify({"error": f"Prediction error: {e}"}), 500

    except Exception as e:
        print(f"❌ Error during prediction: {e}")  # Print error to console
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)  # Allow external access if needed
