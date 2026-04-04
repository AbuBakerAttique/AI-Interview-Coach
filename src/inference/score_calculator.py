"""
Score Fusion — Combines all 4 model outputs into a final interview confidence score.
Weights: Speech 30% | Face 25% | Posture 25% | Eye Contact 20%
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# Score weights (must sum to 1.0)
SCORE_WEIGHTS = {
    "speech":      0.30,
    "face":        0.25,
    "posture":     0.25,
    "eye_contact": 0.20,
}

# Map class index → score contribution
SPEECH_CLASS_SCORES  = {0: 20, 1: 60, 2: 100}   # nervous=20, neutral=60, confident=100
FACE_CLASS_SCORES    = {0: 20, 1: 60, 2: 100}
POSTURE_CLASS_SCORES = {
    "good_posture":  100,
    "leaning_back":   55,
    "crossed_arms":   40,
    "slouching":      30,
    "looking_down":   20,
}


@dataclass
class FrameScores:
    """Scores from a single video frame."""
    speech_class: Optional[int] = None
    speech_confidence: float = 0.0
    face_class: Optional[int] = None
    face_confidence: float = 0.0
    posture_class: Optional[str] = None
    posture_confidence: float = 0.0
    eye_contact_score: float = 0.0


@dataclass
class SessionReport:
    """Final interview session report."""
    final_score: float = 0.0
    speech_avg: float = 0.0
    face_avg: float = 0.0
    posture_avg: float = 0.0
    eye_contact_avg: float = 0.0
    duration_seconds: int = 0
    grade: str = "F"
    feedback: list = field(default_factory=list)


class ScoreCalculator:
    """
    Aggregates per-frame scores across the session and computes the final score.
    """

    def __init__(self):
        self.frame_history: list[FrameScores] = []

    def add_frame(self, scores: FrameScores):
        self.frame_history.append(scores)

    def _get_speech_score(self, frame: FrameScores) -> float:
        if frame.speech_class is None:
            return 50.0  # Default neutral if no audio
        base = SPEECH_CLASS_SCORES.get(frame.speech_class, 50)
        # Weighted by model confidence
        return base * frame.speech_confidence + 50 * (1 - frame.speech_confidence)

    def _get_face_score(self, frame: FrameScores) -> float:
        if frame.face_class is None:
            return 50.0
        base = FACE_CLASS_SCORES.get(frame.face_class, 50)
        return base * frame.face_confidence + 50 * (1 - frame.face_confidence)

    def _get_posture_score(self, frame: FrameScores) -> float:
        if frame.posture_class is None:
            return 50.0
        return POSTURE_CLASS_SCORES.get(frame.posture_class, 50)

    def compute_session_report(self, duration_seconds: int = 0) -> SessionReport:
        if not self.frame_history:
            return SessionReport()

        speech_scores   = [self._get_speech_score(f) for f in self.frame_history]
        face_scores     = [self._get_face_score(f) for f in self.frame_history]
        posture_scores  = [self._get_posture_score(f) for f in self.frame_history]
        eye_scores      = [f.eye_contact_score for f in self.frame_history]

        speech_avg      = np.mean(speech_scores)
        face_avg        = np.mean(face_scores)
        posture_avg     = np.mean(posture_scores)
        eye_avg         = np.mean(eye_scores)

        final_score = (
            speech_avg  * SCORE_WEIGHTS["speech"] +
            face_avg    * SCORE_WEIGHTS["face"] +
            posture_avg * SCORE_WEIGHTS["posture"] +
            eye_avg     * SCORE_WEIGHTS["eye_contact"]
        )

        report = SessionReport(
            final_score=round(final_score, 1),
            speech_avg=round(speech_avg, 1),
            face_avg=round(face_avg, 1),
            posture_avg=round(posture_avg, 1),
            eye_contact_avg=round(eye_avg, 1),
            duration_seconds=duration_seconds,
            grade=self._get_grade(final_score),
            feedback=self._generate_feedback(speech_avg, face_avg, posture_avg, eye_avg),
        )

        return report

    def _get_grade(self, score: float) -> str:
        if score >= 85: return "A"
        if score >= 70: return "B"
        if score >= 55: return "C"
        if score >= 40: return "D"
        return "F"

    def _generate_feedback(self, speech, face, posture, eye) -> list:
        feedback = []

        if speech < 50:
            feedback.append("Your speech tone sounded nervous. Try to speak slower and with more confidence.")
        elif speech >= 80:
            feedback.append("Great speech confidence! Your tone was strong and clear.")

        if face < 50:
            feedback.append("Your facial expressions showed stress. Try to relax your face and maintain a calm expression.")
        elif face >= 80:
            feedback.append("Excellent facial composure throughout the interview.")

        if posture < 50:
            feedback.append("Work on your posture. Sit upright with shoulders back — it signals confidence.")
        elif posture >= 80:
            feedback.append("Your posture was strong and professional.")

        if eye < 50:
            feedback.append("Maintain more eye contact with the camera. Looking away frequently signals nervousness.")
        elif eye >= 80:
            feedback.append("Great eye contact maintained throughout!")

        return feedback

    def reset(self):
        self.frame_history.clear()
