from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import os

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")

app = Flask(__name__, static_folder="Frontend", static_url_path="/")

# simple in-memory test login (replace with real auth in production)
USERS = {"admin@gmail.com": "admin@1234"}

def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Model or scaler not found. Run training first.")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/quiz")
def quiz_page():
    return send_from_directory(app.static_folder, "quiz.html")

@app.route("/dashboard")
def dashboard_page():
    return send_from_directory(app.static_folder, "dashboard.html")

@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.json or {}
    email = data.get("email")
    password = data.get("password")
    if email in USERS and USERS[email] == password:
        return jsonify({"ok": True, "message": "Logged in", "token": "fake-jwt-token-for-demo"})
    return jsonify({"ok": False, "message": "Invalid credentials"}), 401

@app.route("/api/recommend", methods=["POST"])
def recommend():
    body = request.json or {}
    # Expected 'answers' as array in the same order as FEATURE_COLUMNS in train.py
    answers = body.get("answers")
    if not isinstance(answers, list) or len(answers) != 11:
        return jsonify({"ok": False, "message": "Invalid answers payload. Expected 11 encoded features."}), 400
    try:
        model, scaler = load_model()
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 500
    arr = np.array([answers], dtype=float)
    arr_scaled = scaler.transform(arr)
    pred = model.predict(arr_scaled)
    # Optionally return top K neighbors or probabilities if model supports predict_proba (KNN does with labels)
    # For simplicity return the predicted label
    return jsonify({"ok": True, "recommendation": pred[0]})

if __name__== "__main__":
    app.run(debug=True, port=5000)