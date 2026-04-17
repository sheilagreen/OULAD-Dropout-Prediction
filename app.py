"""
Flask API for Student Dropout Early-Warning Prediction

Serves the trained Balanced Random Forest model for real-time
dropout risk scoring. Accepts student features as JSON and returns
a dropout probability, risk tier, and recommended threshold.

Usage:
    python app.py

Endpoints:
    POST /predict  — Score a single student record
    GET  /health   — Health check
"""

import os
import pandas as pd
import joblib
from flask import Flask, request, jsonify

# ── Configuration ──
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "early_warning_rf_balanced.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "model_features.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

app = Flask(__name__)

# ── Load model artifacts ──
try:
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model artifacts loaded successfully.")
except FileNotFoundError as e:
    print(f"Warning: Could not load model artifacts — {e}")
    print("Run the notebook first to generate model files in models/")
    model = None
    features = None
    scaler = None


def prepare_input(data):
    """
    Convert raw JSON input to the feature matrix expected by the model.

    Handles one-hot encoding alignment so that missing categorical
    columns are filled with 0 and extra columns are dropped.
    """
    input_df = pd.DataFrame([data])
    input_encoded = pd.get_dummies(input_df, drop_first=True)

    # Align columns with training feature set
    for col in features:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    return input_encoded[features]


@app.route("/predict", methods=["POST"])
def predict():
    """
    Score a single student record.

    Expects JSON body with student features. Returns:
        - dropout_probability: float (0–1)
        - risk_tier: High / Medium / Low
        - threshold_used: 0.30 (recommended operating point)
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Run notebook first."}), 503

    try:
        data = request.get_json()
        df_in = prepare_input(data)
        prob = model.predict_proba(df_in)[:, 1][0]
        tier = "High" if prob >= 0.65 else "Medium" if prob >= 0.40 else "Low"

        return jsonify({
            "dropout_probability": round(float(prob), 4),
            "risk_tier": tier,
            "threshold_used": 0.30,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "model": "BalancedRandomForest (tuned)",
        "model_loaded": model is not None,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
