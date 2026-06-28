from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image


ROOT = Path(__file__).parent
METRICS_DIR = ROOT / "reports" / "model_metrics"
ASSETS_DIR = ROOT / "assets"
SAVED_MODELS_DIR = ROOT / "saved_models"
FACE_CLASSES = ["nervous", "neutral", "confident"]
CLASS_SCORES = {"nervous": 20, "neutral": 60, "confident": 100}
POSTURE_SCORES = {
    "good_posture": 100,
    "leaning_back": 55,
    "crossed_arms": 40,
    "slouching": 30,
    "looking_down": 20,
}
SCORE_WEIGHTS = {
    "speech": 0.30,
    "face": 0.25,
    "posture": 0.25,
    "eye_contact": 0.20,
}


st.set_page_config(
    page_title="AI Interview Coach",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def percent(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def render_html_demo() -> None:
    html_path = ROOT / "ai_interview_coach_live_app.html"
    if not html_path.exists():
        st.warning("Live demo HTML file was not found.")
        return

    demo_html = html_path.read_text(encoding="utf-8")
    components.html(demo_html, height=470, scrolling=False)


@st.cache_resource
def load_face_classifier():
    import torch
    from src.models.face_model import build_face_model

    model_path = SAVED_MODELS_DIR / "facial_expression" / "face_model.pth"
    if not model_path.exists():
        return None, "Face model weights were not found."

    model = build_face_model(num_classes=len(FACE_CLASSES))
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model, None


def predict_face(image: Image.Image) -> tuple[str, float, dict[str, float]]:
    import torch

    model, error = load_face_classifier()
    if error or model is None:
        raise RuntimeError(error or "Face model could not be loaded.")

    gray = image.convert("L").resize((48, 48))
    image_array = np.asarray(gray, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    probability_map = {
        label: float(probability)
        for label, probability in zip(FACE_CLASSES, probabilities, strict=True)
    }
    prediction = max(probability_map, key=probability_map.get)
    return prediction, probability_map[prediction], probability_map


def grade(score: float) -> str:
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 55:
        return "C"
    if score >= 40:
        return "D"
    return "F"


def weighted_score(
    speech_class: str,
    face_class: str,
    posture_class: str,
    eye_contact: int,
) -> float:
    return (
        CLASS_SCORES[speech_class] * SCORE_WEIGHTS["speech"]
        + CLASS_SCORES[face_class] * SCORE_WEIGHTS["face"]
        + POSTURE_SCORES[posture_class] * SCORE_WEIGHTS["posture"]
        + eye_contact * SCORE_WEIGHTS["eye_contact"]
    )


speech_metrics = load_json(METRICS_DIR / "speech_metrics.json")
face_metrics = load_json(METRICS_DIR / "face_metrics.json")
posture_metrics = load_json(METRICS_DIR / "posture_metrics.json")


st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 3rem; }
      [data-testid="stMetricValue"] { font-size: 1.8rem; }
      .section-copy { color: #5f6b7a; font-size: 0.98rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("AI Interview Coach")
st.caption("Real-time multi-modal interview performance analysis")

overview_left, overview_right = st.columns([1.15, 0.85], gap="large")

with overview_left:
    st.subheader("Live Interface Preview")
    render_html_demo()

with overview_right:
    st.subheader("Session Score")
    score_cols = st.columns(2)
    score_cols[0].metric("Final Score", "78.4 / 100", "Grade B")
    score_cols[1].metric("Eye Contact", "72 / 100", "+8")

    st.progress(0.784)
    st.write(
        "The system combines speech, facial expression, posture, and eye-contact signals "
        "into a single interview confidence score with actionable feedback."
    )

    st.info(
        "This Streamlit version is deployment-ready as a hosted project showcase. "
        "The native real-time webcam and microphone pipeline remains available locally "
        "through `src/inference/real_time_analyzer.py`."
    )


st.divider()

st.subheader("Model Test Lab")
st.write(
    "Upload a face image to run the trained facial-expression model from "
    "`saved_models/facial_expression/face_model.pth`. The prediction is then used in "
    "the same weighted interview score formula."
)

model_cols = st.columns(3)
model_files = {
    "Speech model": SAVED_MODELS_DIR / "speech_confidence" / "speech_model.pth",
    "Face model": SAVED_MODELS_DIR / "facial_expression" / "face_model.pth",
    "Posture model": SAVED_MODELS_DIR / "posture_scorer" / "posture_model.pth",
}
for column, (label, path) in zip(model_cols, model_files.items(), strict=True):
    column.metric(label, "Included" if path.exists() else "Missing")

tester_left, tester_right = st.columns([0.85, 1.15], gap="large")

with tester_left:
    uploaded_image = st.file_uploader(
        "Upload a face image",
        type=["jpg", "jpeg", "png"],
    )
    speech_choice = st.selectbox("Speech confidence", FACE_CLASSES, index=2)
    posture_choice = st.selectbox("Posture", list(POSTURE_SCORES), index=0)
    eye_contact = st.slider("Eye contact score", min_value=0, max_value=100, value=72)

with tester_right:
    face_choice = "neutral"
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded image", width=260)
        try:
            face_choice, confidence, probabilities = predict_face(image)
            st.success(f"Face model prediction: {face_choice} ({confidence:.1%})")
            for label, probability in probabilities.items():
                st.progress(probability, text=f"{label}: {probability:.1%}")
        except Exception as exc:
            st.error(f"Could not run the face model: {exc}")
    else:
        st.info("Upload an image to run the real face model. Until then, face defaults to neutral.")

    final_score = weighted_score(
        speech_class=speech_choice,
        face_class=face_choice,
        posture_class=posture_choice,
        eye_contact=eye_contact,
    )
    score_a, score_b = st.columns(2)
    score_a.metric("Computed Score", f"{final_score:.1f} / 100")
    score_b.metric("Grade", grade(final_score))


st.divider()

st.subheader("Model Performance")
metric_cols = st.columns(4)
metric_cols[0].metric(
    "Speech Confidence",
    percent(speech_metrics.get("best_val_accuracy")),
    f"{speech_metrics.get('epochs_trained', 'N/A')} epochs",
)
metric_cols[1].metric(
    "Facial Expression",
    percent(face_metrics.get("best_val_accuracy")),
    f"{face_metrics.get('epochs_trained', 'N/A')} epochs",
)
metric_cols[2].metric(
    "Posture Scorer",
    percent(posture_metrics.get("best_val_accuracy")),
    f"{posture_metrics.get('epochs_trained', 'N/A')} epochs",
)
metric_cols[3].metric("Eye Contact", "Rule-based", "MediaPipe geometry")

curve_tabs = st.tabs(["Speech", "Face", "Posture"])
curve_files = {
    "Speech": METRICS_DIR / "speech_training_curve.png",
    "Face": METRICS_DIR / "face_training_curve.png",
    "Posture": METRICS_DIR / "posture_training_curve.png",
}

for tab, (label, image_path) in zip(curve_tabs, curve_files.items(), strict=True):
    with tab:
        if image_path.exists():
            st.image(str(image_path), caption=f"{label} training curve", use_container_width=True)
        else:
            st.warning(f"{label} training curve image was not found.")


st.divider()

details_left, details_right = st.columns([0.95, 1.05], gap="large")

with details_left:
    st.subheader("Architecture")
    architecture = ASSETS_DIR / "architecture_diagram.png"
    if architecture.exists():
        st.image(str(architecture), use_container_width=True)
    else:
        st.write("Architecture diagram missing from `assets/`.")

with details_right:
    st.subheader("How It Works")
    st.markdown(
        """
        - **Speech:** CNN + LSTM classifier over MFCC audio features.
        - **Face:** Custom CNN for nervous, neutral, and confident expression classes.
        - **Posture:** Neural network over body landmark features.
        - **Eye contact:** Geometric gaze estimate from face mesh landmarks.
        - **Fusion:** Weighted score calculator produces the final grade and feedback.
        """
    )

    st.code(
        "streamlit run streamlit_app.py",
        language="bash",
    )
