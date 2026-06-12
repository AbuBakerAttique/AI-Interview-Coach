"""
Real-Time Interview Analyzer
Combines all 4 models into one live webcam system.
Run: python src/inference/real_time_analyzer.py
"""

import cv2
import numpy as np
import torch
import mediapipe as mp
import librosa
import sounddevice as sd
import threading
import queue
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.speech_model import build_speech_model
from src.models.face_model import build_face_model
from src.models.posture_model import build_posture_model
from src.models.eye_contact import EyeContactDetector
from src.inference.score_calculator import ScoreCalculator, FrameScores

# --- Config ---
SPEECH_MODEL_PATH  = "saved_models/speech_confidence/speech_model.pth"
FACE_MODEL_PATH    = "saved_models/facial_expression/face_model.pth"
POSTURE_MODEL_PATH = "saved_models/posture_scorer/posture_model.pth"

N_MFCC   = 40
MAX_LEN  = 200
IMG_SIZE = 48

POSTURE_CLASSES = ['TLB', 'TLF', 'TLL', 'TLR', 'TUP']
POSTURE_LABELS  = {
    'TUP': 'good_posture',
    'TLF': 'slouching',
    'TLB': 'leaning_back',
    'TLL': 'leaning_left',
    'TLR': 'leaning_right',
}

CLASS_NAMES = {
    "speech":  {0: "Nervous", 1: "Neutral", 2: "Confident"},
    "face":    {0: "Nervous", 1: "Neutral", 2: "Confident"},
}

COLORS = {
    "Confident": (0, 255, 0),    # Green
    "Neutral":   (255, 165, 0),  # Orange
    "Nervous":   (0, 0, 255),    # Red
}


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_speech_model():
    checkpoint = torch.load(SPEECH_MODEL_PATH, map_location="cpu", weights_only=False)
    model = build_speech_model(num_classes=3)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_face_model():
    checkpoint = torch.load(FACE_MODEL_PATH, map_location="cpu", weights_only=False)
    model = build_face_model(num_classes=3)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_posture_model():
    checkpoint = torch.load(POSTURE_MODEL_PATH, map_location="cpu", weights_only=False)
    model = build_posture_model(input_dim=99, num_classes=5)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


# ─── Audio Thread ─────────────────────────────────────────────────────────────

class AudioAnalyzer(threading.Thread):
    """
    Runs in background thread.
    Continuously records 3-second audio chunks and predicts speech confidence.
    """

    def __init__(self, model, result_queue, sample_rate=22050, chunk_duration=3):
        super().__init__(daemon=True)
        self.model        = model
        self.result_queue = result_queue
        self.sample_rate  = sample_rate
        self.chunk_dur    = chunk_duration
        self.running      = True

    def extract_mfcc(self, audio):
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=N_MFCC).T
        if mfcc.shape[0] < MAX_LEN:
            mfcc = np.pad(mfcc, ((0, MAX_LEN - mfcc.shape[0]), (0, 0)))
        else:
            mfcc = mfcc[:MAX_LEN, :]
        return mfcc.astype(np.float32)

    def run(self):
        while self.running:
            try:
                audio = sd.rec(
                    int(self.chunk_dur * self.sample_rate),
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="float32"
                )
                sd.wait()
                audio = audio.flatten()

                mfcc = self.extract_mfcc(audio)
                tensor = torch.tensor(mfcc).unsqueeze(0)  # (1, 200, 40)

                with torch.no_grad():
                    output = self.model(tensor)
                    probs  = torch.softmax(output, dim=1)[0]
                    pred   = int(probs.argmax())
                    conf   = float(probs.max())

                self.result_queue.put({"speech_class": pred, "speech_confidence": conf})

            except Exception as e:
                pass

    def stop(self):
        self.running = False


# ─── Main Analyzer ────────────────────────────────────────────────────────────

class RealTimeAnalyzer:

    def __init__(self):
        print("Loading models...")
        self.speech_model  = load_speech_model()
        self.face_model    = load_face_model()
        self.posture_model = load_posture_model()
        self.eye_detector  = EyeContactDetector()
        self.score_calc    = ScoreCalculator()
        print("All models loaded!\n")

        # MediaPipe
        self.mp_pose    = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose       = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # Face detector for cropping
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Audio queue
        self.audio_queue  = queue.Queue(maxsize=1)
        self.audio_thread = AudioAnalyzer(self.speech_model, self.audio_queue)

        # State
        self.current_speech = {"speech_class": 1, "speech_confidence": 0.5}
        self.session_active = False
        self.start_time     = None

    def predict_face(self, frame):
        """Detect face and predict expression."""
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return None, 0.0

        x, y, w, h = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
        face_roi = face_roi.astype(np.float32) / 255.0

        tensor = torch.tensor(face_roi).unsqueeze(0).unsqueeze(0)  # (1,1,48,48)

        with torch.no_grad():
            output = self.face_model(tensor)
            probs  = torch.softmax(output, dim=1)[0]
            pred   = int(probs.argmax())
            conf   = float(probs.max())

        return pred, conf

    def predict_posture(self, results):
        """Predict posture from MediaPipe landmarks."""
        if not results.pose_landmarks:
            return None, 0.0

        landmarks = results.pose_landmarks.landmark
        features  = []
        for lm in landmarks:
            features.extend([lm.x, lm.y, lm.z])

        tensor = torch.tensor([features], dtype=torch.float32)

        with torch.no_grad():
            output = self.posture_model(tensor)
            probs  = torch.softmax(output, dim=1)[0]
            pred   = int(probs.argmax())
            conf   = float(probs.max())

        posture_code  = POSTURE_CLASSES[pred]
        posture_label = POSTURE_LABELS.get(posture_code, "unknown")
        return posture_label, conf

    def draw_overlay(self, frame, face_class, face_conf, posture_label, eye_result):
        """Draw scores and info on the video frame."""
        h, w = frame.shape[:2]

        # Background panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (320, 220), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Speech
        speech_label = CLASS_NAMES["speech"].get(self.current_speech["speech_class"], "Neutral")
        speech_color = COLORS.get(speech_label, (255, 255, 255))
        cv2.putText(frame, f"Speech:  {speech_label}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, speech_color, 2)

        # Face
        if face_class is not None:
            face_label = CLASS_NAMES["face"].get(face_class, "Neutral")
            face_color = COLORS.get(face_label, (255, 255, 255))
            cv2.putText(frame, f"Face:    {face_label}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, face_color, 2)

        # Posture
        if posture_label:
            posture_color = (0, 255, 0) if posture_label == "good_posture" else (0, 0, 255)
            cv2.putText(frame, f"Posture: {posture_label}", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, posture_color, 2)

        # Eye contact
        eye_score = eye_result.get("score", 0)
        eye_color = (0, 255, 0) if eye_score > 60 else (0, 0, 255)
        cv2.putText(frame, f"Eye:     {eye_score:.0f}/100", (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, eye_color, 2)

        # Session timer
        if self.session_active and self.start_time:
            elapsed = int(time.time() - self.start_time)
            mins, secs = divmod(elapsed, 60)
            cv2.putText(frame, f"Session: {mins:02d}:{secs:02d}", (10, 165),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

        # Controls
        cv2.putText(frame, "SPACE: Start/Stop | Q: Quit & Report", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Recording indicator
        if self.session_active:
            cv2.circle(frame, (w - 20, 20), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (w - 55, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return frame

    def print_report(self):
        """Print final session report."""
        duration = int(time.time() - self.start_time) if self.start_time else 0
        report   = self.score_calc.compute_session_report(duration_seconds=duration)

        print("\n" + "="*50)
        print("      INTERVIEW SESSION REPORT")
        print("="*50)
        print(f"  Final Score   : {report.final_score}/100")
        print(f"  Grade         : {report.grade}")
        print(f"  Duration      : {duration//60:02d}:{duration%60:02d}")
        print("-"*50)
        print(f"  Speech        : {report.speech_avg}/100")
        print(f"  Face          : {report.face_avg}/100")
        print(f"  Posture       : {report.posture_avg}/100")
        print(f"  Eye Contact   : {report.eye_contact_avg}/100")
        print("-"*50)
        print("  Feedback:")
        for tip in report.feedback:
            print(f"  • {tip}")
        print("="*50 + "\n")

    def run(self):
        """Main loop."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        self.audio_thread.start()
        print("System ready!")
        print("Press SPACE to start recording your session.")
        print("Press Q to quit and see your report.\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Pose detection
            pose_results = self.pose.process(rgb)
            if pose_results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, pose_results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )

            # Get latest audio result
            try:
                self.current_speech = self.audio_queue.get_nowait()
            except queue.Empty:
                pass

            # Predictions
            face_class, face_conf     = self.predict_face(frame)
            posture_label, post_conf  = self.predict_posture(pose_results)
            eye_result                = self.eye_detector.process_frame(frame)

            # Record scores if session active
            if self.session_active:
                scores = FrameScores(
                    speech_class=self.current_speech["speech_class"],
                    speech_confidence=self.current_speech["speech_confidence"],
                    face_class=face_class,
                    face_confidence=face_conf,
                    posture_class=posture_label,
                    posture_confidence=post_conf,
                    eye_contact_score=eye_result.get("score", 0),
                )
                self.score_calc.add_frame(scores)

            # Draw overlay
            frame = self.draw_overlay(frame, face_class, face_conf, posture_label, eye_result)

            cv2.imshow("AI Interview Coach", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                if self.session_active:
                    self.print_report()
                break
            elif key == ord(" "):
                self.session_active = not self.session_active
                if self.session_active:
                    self.start_time = time.time()
                    self.score_calc.reset()
                    print("Session started! Recording your interview...")
                else:
                    self.print_report()
                    print("Session paused. Press SPACE to start a new session.")

        cap.release()
        self.audio_thread.stop()
        self.eye_detector.release()
        self.pose.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    analyzer = RealTimeAnalyzer()
    analyzer.run()
