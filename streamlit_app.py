from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


ROOT = Path(__file__).parent
METRICS_DIR = ROOT / "reports" / "model_metrics"
ASSETS_DIR = ROOT / "assets"


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

