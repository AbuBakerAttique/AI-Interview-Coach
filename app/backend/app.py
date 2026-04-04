"""
Flask API — AI Interview Coach Backend
Serves real-time predictions via REST endpoints.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import base64
import cv2

app = Flask(__name__)
CORS(app)

# Models are loaded once at startup
speech_model = None
face_model   = None
posture_model = None


def load_models():
    """Load all trained models at startup."""
    import tensorflow as tf
    global speech_model, face_model, posture_model

    try:
        speech_model  = tf.keras.models.load_model("../../saved_models/speech_confidence/speech_model.keras")
        face_model    = tf.keras.models.load_model("../../saved_models/facial_expression/face_model.keras")
        posture_model = tf.keras.models.load_model("../../saved_models/posture_scorer/posture_model.keras")
        print("All models loaded successfully.")
    except Exception as e:
        print(f"Model loading error: {e}")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_loaded": speech_model is not None})


@app.route("/analyze/frame", methods=["POST"])
def analyze_frame():
    """
    Analyze a single video frame.
    Expects: { "image": "<base64 encoded jpg>" }
    Returns: { "face_class": int, "face_confidence": float, "posture_class": str }
    """
    data = request.json
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    # Decode base64 image
    img_bytes = base64.b64decode(data["image"])
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    result = {}

    # Face analysis
    if face_model is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_img = cv2.resize(gray, (48, 48)).astype(np.float32) / 255.0
        face_img = face_img.reshape(1, 48, 48, 1)
        pred = face_model.predict(face_img, verbose=0)[0]
        result["face_class"] = int(np.argmax(pred))
        result["face_confidence"] = float(np.max(pred))

    return jsonify(result)


@app.route("/analyze/audio", methods=["POST"])
def analyze_audio():
    """
    Analyze an audio chunk for speech confidence.
    Expects: { "mfcc": [[...]] }  — MFCC features as 2D list
    Returns: { "speech_class": int, "speech_confidence": float }
    """
    data = request.json
    if not data or "mfcc" not in data:
        return jsonify({"error": "No MFCC data provided"}), 400

    mfcc = np.array(data["mfcc"]).reshape(1, 200, 40)

    if speech_model is not None:
        pred = speech_model.predict(mfcc, verbose=0)[0]
        return jsonify({
            "speech_class": int(np.argmax(pred)),
            "speech_confidence": float(np.max(pred))
        })

    return jsonify({"speech_class": 1, "speech_confidence": 0.5})


@app.route("/report", methods=["POST"])
def generate_report():
    """
    Generate a final session report.
    Expects: { "scores": { "speech": float, "face": float, "posture": float, "eye_contact": float } }
    """
    data = request.json
    scores = data.get("scores", {})

    weights = {"speech": 0.30, "face": 0.25, "posture": 0.25, "eye_contact": 0.20}
    final_score = sum(scores.get(k, 50) * w for k, w in weights.items())

    grade = "A" if final_score >= 85 else "B" if final_score >= 70 else "C" if final_score >= 55 else "D" if final_score >= 40 else "F"

    return jsonify({
        "final_score": round(final_score, 1),
        "grade": grade,
        "breakdown": scores,
    })


if __name__ == "__main__":
    load_models()
    app.run(debug=True, host="0.0.0.0", port=5000)
