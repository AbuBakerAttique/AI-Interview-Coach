from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Lock

import numpy as np
import streamlit as st
from PIL import Image

from src.inference.score_calculator import FrameScores, ScoreCalculator


ROOT = Path(__file__).parent
METRICS_DIR = ROOT / "reports" / "model_metrics"
ASSETS_DIR = ROOT / "assets"
SAVED_MODELS_DIR = ROOT / "saved_models"
FACE_CLASSES = ["nervous", "neutral", "confident"]
CLASS_SCORES = {"nervous": 20, "neutral": 60, "confident": 100}
POSTURE_SCORES = {
    "good_posture": 100,
    "leaning_back": 55,
    "leaning_left": 55,
    "leaning_right": 55,
    "crossed_arms": 40,
    "slouching": 30,
    "looking_down": 20,
}
FACE_INDEX_TO_LABEL = {0: "Nervous", 1: "Neutral", 2: "Confident"}
POSTURE_LABELS = {
    "TUP": "good_posture",
    "TLF": "slouching",
    "TLB": "leaning_back",
    "TLL": "leaning_left",
    "TLR": "leaning_right",
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


@st.cache_resource
def load_live_models():
    import cv2
    import mediapipe as mp
    import torch
    from src.models.face_model import build_face_model
    from src.models.posture_model import POSTURE_CLASSES as DEFAULT_POSTURE_CLASSES
    from src.models.posture_model import build_posture_model
    from src.models.speech_model import build_speech_model

    face_path = SAVED_MODELS_DIR / "facial_expression" / "face_model.pth"
    posture_path = SAVED_MODELS_DIR / "posture_scorer" / "posture_model.pth"
    speech_path = SAVED_MODELS_DIR / "speech_confidence" / "speech_model.pth"

    face_model = build_face_model(num_classes=len(FACE_CLASSES))
    face_checkpoint = torch.load(face_path, map_location="cpu")
    face_state = (
        face_checkpoint.get("model_state_dict", face_checkpoint)
        if isinstance(face_checkpoint, dict)
        else face_checkpoint
    )
    face_model.load_state_dict(face_state)
    face_model.eval()

    posture_checkpoint = torch.load(posture_path, map_location="cpu", weights_only=False)
    posture_state = (
        posture_checkpoint.get("model_state_dict", posture_checkpoint)
        if isinstance(posture_checkpoint, dict)
        else posture_checkpoint
    )
    posture_classes = (
        list(posture_checkpoint.get("label_encoder_classes"))
        if isinstance(posture_checkpoint, dict) and posture_checkpoint.get("label_encoder_classes") is not None
        else DEFAULT_POSTURE_CLASSES
    )
    posture_model = build_posture_model(
        input_dim=posture_checkpoint.get("input_dim", 99) if isinstance(posture_checkpoint, dict) else 99,
        num_classes=posture_checkpoint.get("num_classes", len(posture_classes)) if isinstance(posture_checkpoint, dict) else 5,
    )
    posture_model.load_state_dict(posture_state)
    posture_model.eval()

    speech_model = build_speech_model(num_classes=len(FACE_CLASSES))
    speech_checkpoint = torch.load(speech_path, map_location="cpu", weights_only=False)
    speech_state = (
        speech_checkpoint.get("model_state_dict", speech_checkpoint)
        if isinstance(speech_checkpoint, dict)
        else speech_checkpoint
    )
    speech_model.load_state_dict(speech_state)
    speech_model.eval()

    return {
        "cv2": cv2,
        "mp": mp,
        "torch": torch,
        "face_model": face_model,
        "posture_model": posture_model,
        "posture_classes": posture_classes,
        "speech_model": speech_model,
    }


class BrowserInterviewSession:
    def __init__(self):
        self.lock = Lock()
        self.score_calc = ScoreCalculator()
        self.recording = False
        self.started_at: float | None = None
        self.final_report = None
        self.latest = {
            "speech_class": 1,
            "speech_confidence": 0.5,
            "face_class": None,
            "face_confidence": 0.0,
            "posture_class": None,
            "posture_confidence": 0.0,
            "eye_contact_score": 0.0,
            "frames": 0,
        }

    def start(self) -> None:
        with self.lock:
            self.score_calc.reset()
            self.recording = True
            self.started_at = time.time()
            self.final_report = None

    def stop(self):
        with self.lock:
            duration = int(time.time() - self.started_at) if self.started_at else 0
            self.recording = False
            self.final_report = self.score_calc.compute_session_report(duration_seconds=duration)
            return self.final_report

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "recording": self.recording,
                "started_at": self.started_at,
                "latest": dict(self.latest),
                "frame_count": len(self.score_calc.frame_history),
                "final_report": self.final_report,
            }


@st.cache_resource
def get_browser_session() -> BrowserInterviewSession:
    return BrowserInterviewSession()


def predict_live_face(frame: np.ndarray, model, face_cascade, torch_module, cv2_module):
    cv2 = cv2_module
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None, 0.0

    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
    face_roi = gray[y : y + h, x : x + w]
    face_roi = cv2.resize(face_roi, (48, 48)).astype(np.float32) / 255.0
    tensor = torch_module.from_numpy(face_roi).unsqueeze(0).unsqueeze(0)

    with torch_module.no_grad():
        output = model(tensor)
        probabilities = torch_module.softmax(output, dim=1)[0]
        prediction = int(probabilities.argmax())
        confidence = float(probabilities.max())

    return prediction, confidence


def predict_live_posture(pose_results, model, posture_classes, torch_module):
    if not pose_results.pose_landmarks:
        return None, 0.0

    features = []
    for landmark in pose_results.pose_landmarks.landmark:
        features.extend([landmark.x, landmark.y, landmark.z])

    tensor = torch_module.tensor([features], dtype=torch_module.float32)
    with torch_module.no_grad():
        output = model(tensor)
        probabilities = torch_module.softmax(output, dim=1)[0]
        prediction = int(probabilities.argmax())
        confidence = float(probabilities.max())

    posture_code = posture_classes[prediction]
    return POSTURE_LABELS.get(posture_code, posture_code), confidence


def extract_speech_mfcc(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    import librosa

    target_rate = 22050
    if sample_rate != target_rate:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_rate)

    mfcc = librosa.feature.mfcc(y=audio, sr=target_rate, n_mfcc=40).T
    if mfcc.shape[0] < 200:
        mfcc = np.pad(mfcc, ((0, 200 - mfcc.shape[0]), (0, 0)))
    else:
        mfcc = mfcc[:200, :]
    return mfcc.astype(np.float32)


class LiveVideoProcessor:
    def __init__(self, session: BrowserInterviewSession):
        from src.models.eye_contact import EyeContactDetector

        self.session = session
        self.resources = load_live_models()
        self.cv2 = self.resources["cv2"]
        self.torch = self.resources["torch"]
        self.pose_api = self.resources["mp"].solutions.pose
        self.drawing = self.resources["mp"].solutions.drawing_utils
        self.pose = self.pose_api.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.face_cascade = self.cv2.CascadeClassifier(
            self.cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_detector = EyeContactDetector()

    def recv(self, frame):
        import av

        image = frame.to_ndarray(format="bgr24")
        image = self.cv2.flip(image, 1)
        rgb = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2RGB)

        pose_results = self.pose.process(rgb)
        if pose_results.pose_landmarks:
            self.drawing.draw_landmarks(
                image,
                pose_results.pose_landmarks,
                self.pose_api.POSE_CONNECTIONS,
            )

        face_class, face_confidence = predict_live_face(
            image,
            self.resources["face_model"],
            self.face_cascade,
            self.torch,
            self.cv2,
        )
        posture_class, posture_confidence = predict_live_posture(
            pose_results,
            self.resources["posture_model"],
            self.resources["posture_classes"],
            self.torch,
        )
        eye_result = self.eye_detector.process_frame(image)

        with self.session.lock:
            speech_class = self.session.latest["speech_class"]
            speech_confidence = self.session.latest["speech_confidence"]
            self.session.latest.update(
                {
                    "face_class": face_class,
                    "face_confidence": face_confidence,
                    "posture_class": posture_class,
                    "posture_confidence": posture_confidence,
                    "eye_contact_score": eye_result.get("score", 0.0),
                    "frames": self.session.latest["frames"] + 1,
                }
            )
            recording = self.session.recording
            if recording:
                self.session.score_calc.add_frame(
                    FrameScores(
                        speech_class=speech_class,
                        speech_confidence=speech_confidence,
                        face_class=face_class,
                        face_confidence=face_confidence,
                        posture_class=posture_class,
                        posture_confidence=posture_confidence,
                        eye_contact_score=eye_result.get("score", 0.0),
                    )
                )
                started_at = self.session.started_at
                recorded_frames = len(self.session.score_calc.frame_history)
            else:
                started_at = None
                recorded_frames = len(self.session.score_calc.frame_history)

        self.draw_overlay(
            image=image,
            recording=recording,
            started_at=started_at,
            recorded_frames=recorded_frames,
            speech_class=speech_class,
            speech_confidence=speech_confidence,
            face_class=face_class,
            face_confidence=face_confidence,
            posture_class=posture_class,
            posture_confidence=posture_confidence,
            eye_score=eye_result.get("score", 0.0),
        )
        return av.VideoFrame.from_ndarray(image, format="bgr24")

    def draw_overlay(
        self,
        image: np.ndarray,
        recording: bool,
        started_at: float | None,
        recorded_frames: int,
        speech_class: int,
        speech_confidence: float,
        face_class: int | None,
        face_confidence: float,
        posture_class: str | None,
        posture_confidence: float,
        eye_score: float,
    ) -> None:
        h, w = image.shape[:2]
        overlay = image.copy()
        self.cv2.rectangle(overlay, (0, 0), (370, 208), (0, 0, 0), -1)
        self.cv2.addWeighted(overlay, 0.58, image, 0.42, 0, image)

        current_report = report_from_latest(
            {
                "speech_class": speech_class,
                "speech_confidence": speech_confidence,
                "face_class": face_class,
                "face_confidence": face_confidence,
                "posture_class": posture_class,
                "posture_confidence": posture_confidence,
                "eye_contact_score": eye_score,
            }
        )
        speech_label = FACE_INDEX_TO_LABEL.get(speech_class, "Neutral")
        face_label = FACE_INDEX_TO_LABEL.get(face_class, "Searching")
        posture_label = posture_class or "searching"
        status = "REC" if recording else "READY"
        elapsed = int(time.time() - started_at) if recording and started_at else 0
        mins, secs = divmod(elapsed, 60)

        rows = [
            (f"Combined: {current_report.final_score:.1f}/100  Grade {current_report.grade}", (255, 255, 255)),
            (f"Speech:  {speech_label}", (80, 255, 80)),
            (f"Face:    {face_label}", (80, 255, 80) if face_class == 2 else (0, 190, 255)),
            (f"Posture: {posture_label}", (80, 255, 80) if posture_class == "good_posture" else (0, 165, 255)),
            (f"Eye:     {eye_score:.0f}/100", (80, 255, 80) if eye_score >= 60 else (0, 165, 255)),
            (f"Session: {status} {mins:02d}:{secs:02d}  frames:{recorded_frames}", (255, 255, 80)),
        ]
        y = 30
        for text, color in rows:
            self.cv2.putText(image, text, (14, y), self.cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)
            y += 30

        if recording:
            self.cv2.circle(image, (w - 28, 28), 10, (0, 0, 255), -1)
            self.cv2.putText(image, "REC", (w - 78, 34), self.cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 255), 2)

        self.cv2.putText(
            image,
            "Start Stream -> Start Session -> Stop Session for report",
            (14, h - 18),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
        )


class LiveAudioProcessor:
    def __init__(self, session: BrowserInterviewSession):
        self.session = session
        self.resources = load_live_models()
        self.torch = self.resources["torch"]
        self.audio_buffer = np.array([], dtype=np.float32)

    def recv(self, frame):
        sample_rate = frame.sample_rate or 48000

        with self.session.lock:
            recording = self.session.recording
        if not recording:
            self.audio_buffer = np.array([], dtype=np.float32)
            return frame

        audio = frame.to_ndarray()
        if audio.ndim > 1:
            audio = audio.mean(axis=0)
        if np.issubdtype(audio.dtype, np.integer):
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
        else:
            audio = audio.astype(np.float32)

        self.audio_buffer = np.concatenate([self.audio_buffer, audio])
        chunk_size = int(sample_rate * 3)
        if len(self.audio_buffer) < chunk_size:
            return frame

        chunk = self.audio_buffer[:chunk_size]
        self.audio_buffer = self.audio_buffer[chunk_size:]
        try:
            mfcc = extract_speech_mfcc(chunk, sample_rate)
            tensor = self.torch.from_numpy(mfcc).unsqueeze(0)
            with self.torch.no_grad():
                output = self.resources["speech_model"](tensor)
                probabilities = self.torch.softmax(output, dim=1)[0]
                prediction = int(probabilities.argmax())
                confidence = float(probabilities.max())
            with self.session.lock:
                self.session.latest.update(
                    {
                        "speech_class": prediction,
                        "speech_confidence": confidence,
                    }
                )
        except Exception:
            pass

        return frame


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


def report_from_latest(latest: dict):
    calculator = ScoreCalculator()
    calculator.add_frame(
        FrameScores(
            speech_class=latest.get("speech_class"),
            speech_confidence=latest.get("speech_confidence", 0.0),
            face_class=latest.get("face_class"),
            face_confidence=latest.get("face_confidence", 0.0),
            posture_class=latest.get("posture_class"),
            posture_confidence=latest.get("posture_confidence", 0.0),
            eye_contact_score=latest.get("eye_contact_score", 0.0),
        )
    )
    return calculator.compute_session_report()


def render_live_interview_session() -> None:
    try:
        from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer
    except ImportError:
        st.error(
            "Real-time browser mode needs `streamlit-webrtc`. Install the updated "
            "`requirements.txt` and restart Streamlit."
        )
        return

    session = get_browser_session()
    snapshot = session.snapshot()

    st.subheader("Live Interview Session")
    st.write(
        "Start the camera stream, then start a session. The app analyzes the live video "
        "and microphone continuously, combines speech, face, posture, and eye contact into "
        "one score, and produces the final report when you stop."
    )

    control_a, control_b, control_c = st.columns([0.22, 0.22, 0.56])
    if control_a.button("Start Session", disabled=snapshot["recording"], use_container_width=True):
        session.start()
        st.rerun()
    if control_b.button("Stop Session", disabled=not snapshot["recording"], use_container_width=True):
        session.stop()
        st.rerun()
    with control_c:
        state_text = "Recording" if snapshot["recording"] else "Ready"
        st.metric("Session State", state_text, f"{snapshot['frame_count']} scored frames")

    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    webrtc_streamer(
        key="ai-interview-coach-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": True},
        video_processor_factory=lambda: LiveVideoProcessor(session),
        audio_processor_factory=lambda: LiveAudioProcessor(session),
        async_processing=True,
        sendback_audio=False,
    )

    snapshot = session.snapshot()
    latest = snapshot["latest"]
    current_report = report_from_latest(latest)
    st.progress(min(current_report.final_score / 100, 1.0), text="Current combined score")
    live_cols = st.columns(5)
    live_cols[0].metric(
        "Combined",
        f"{current_report.final_score:.1f} / 100",
        f"Grade {current_report.grade}",
    )
    live_cols[1].metric(
        "Speech",
        FACE_INDEX_TO_LABEL.get(latest["speech_class"], "Neutral"),
        f"{latest['speech_confidence']:.0%}",
    )
    live_cols[2].metric(
        "Face",
        FACE_INDEX_TO_LABEL.get(latest["face_class"], "Searching"),
        f"{latest['face_confidence']:.0%}",
    )
    live_cols[3].metric(
        "Posture",
        latest["posture_class"] or "searching",
        f"{latest['posture_confidence']:.0%}",
    )
    live_cols[4].metric("Eye Contact", f"{latest['eye_contact_score']:.0f} / 100")

    if snapshot["final_report"] is not None:
        report = snapshot["final_report"]
        st.success(f"Final Score: {report.final_score:.1f} / 100 · Grade {report.grade}")
        st.progress(min(report.final_score / 100, 1.0))
        report_cols = st.columns(4)
        report_cols[0].metric("Speech Avg", f"{report.speech_avg:.1f}")
        report_cols[1].metric("Face Avg", f"{report.face_avg:.1f}")
        report_cols[2].metric("Posture Avg", f"{report.posture_avg:.1f}")
        report_cols[3].metric("Eye Avg", f"{report.eye_contact_avg:.1f}")
        if report.feedback:
            st.markdown("**Feedback**")
            for item in report.feedback:
                st.write(f"- {item}")


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

render_live_interview_session()


st.divider()

st.subheader("Snapshot Test Lab")
st.write(
    "Use your camera or upload a face image to run the trained facial-expression model from "
    "`saved_models/facial_expression/face_model.pth` without starting a full session."
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
    image_source = st.radio(
        "Face image source",
        ["Live camera", "Upload image"],
        horizontal=True,
    )
    camera_image = None
    uploaded_image = None
    if image_source == "Live camera":
        camera_image = st.camera_input("Take a live face photo")
    else:
        uploaded_image = st.file_uploader(
            "Upload a face image",
            type=["jpg", "jpeg", "png"],
        )
    speech_choice = st.selectbox("Speech confidence", FACE_CLASSES, index=2)
    posture_choice = st.selectbox("Posture", list(POSTURE_SCORES), index=0)
    eye_contact = st.slider("Eye contact score", min_value=0, max_value=100, value=72)

with tester_right:
    face_choice = "neutral"
    selected_image = camera_image or uploaded_image
    selected_caption = "Live camera capture" if camera_image is not None else "Uploaded image"
    if selected_image is not None:
        image = Image.open(selected_image)
        st.image(image, caption=selected_caption, width=260)
        try:
            face_choice, confidence, probabilities = predict_face(image)
            st.success(f"Face model prediction: {face_choice} ({confidence:.1%})")
            for label, probability in probabilities.items():
                st.progress(probability, text=f"{label}: {probability:.1%}")
        except Exception as exc:
            st.error(f"Could not run the face model: {exc}")
    else:
        st.info("Capture or upload an image to run the real face model. Until then, face defaults to neutral.")

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
