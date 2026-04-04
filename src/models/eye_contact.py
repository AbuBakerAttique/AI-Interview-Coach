"""
Eye Contact Detector — MediaPipe Face Mesh + Gaze Ratio
No external dataset needed. Uses iris landmark geometry.
"""

import cv2
import mediapipe as mp
import numpy as np


mp_face_mesh = mp.solutions.face_mesh

# MediaPipe Face Mesh landmark indices for eyes and iris
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473


class EyeContactDetector:
    """
    Detects whether the candidate is making eye contact with the camera.
    Uses iris landmark ratios from MediaPipe Face Mesh.

    Eye contact score: 0 (looking away) → 100 (direct camera contact)
    """

    def __init__(self, min_detection_confidence: float = 0.7):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Enables iris tracking (landmarks 468-477)
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        self.score_history = []
        self.history_window = 30  # Smooth over 30 frames

    def get_gaze_ratio(self, landmarks, eye_indices: list, iris_center_idx: int, frame_w: int, frame_h: int) -> float:
        """
        Compute gaze ratio for one eye.
        Ratio = iris_x position relative to eye corners.
        Ratio ~0.5 = looking straight → eye contact
        Ratio <0.35 or >0.65 = looking sideways
        """
        eye_points = np.array([
            [landmarks[i].x * frame_w, landmarks[i].y * frame_h]
            for i in eye_indices
        ])

        iris = np.array([
            landmarks[iris_center_idx].x * frame_w,
            landmarks[iris_center_idx].y * frame_h
        ])

        # Eye bounding box
        min_x = eye_points[:, 0].min()
        max_x = eye_points[:, 0].max()

        if max_x == min_x:
            return 0.5

        # Normalized iris position in horizontal range
        gaze_ratio = (iris[0] - min_x) / (max_x - min_x)
        return gaze_ratio

    def compute_eye_contact_score(self, left_ratio: float, right_ratio: float) -> float:
        """
        Convert gaze ratios to eye contact score (0-100).
        Both eyes should have ratio near 0.5 for direct camera contact.
        """
        center_threshold = 0.15  # Acceptable deviation from 0.5

        left_deviation = abs(left_ratio - 0.5)
        right_deviation = abs(right_ratio - 0.5)
        avg_deviation = (left_deviation + right_deviation) / 2

        # Score: 100 when deviation=0, 0 when deviation >= threshold
        score = max(0.0, 1.0 - (avg_deviation / center_threshold)) * 100
        return min(100.0, score)

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process a single video frame and return eye contact score.

        Args:
            frame: BGR frame from OpenCV

        Returns:
            dict with keys: score, left_ratio, right_ratio, eye_contact_detected
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return {"score": 0.0, "eye_contact_detected": False}

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        try:
            left_ratio = self.get_gaze_ratio(landmarks, LEFT_EYE_INDICES, LEFT_IRIS_CENTER, w, h)
            right_ratio = self.get_gaze_ratio(landmarks, RIGHT_EYE_INDICES, RIGHT_IRIS_CENTER, w, h)
            score = self.compute_eye_contact_score(left_ratio, right_ratio)
        except Exception:
            return {"score": 0.0, "eye_contact_detected": False}

        # Smooth score over recent frames
        self.score_history.append(score)
        if len(self.score_history) > self.history_window:
            self.score_history.pop(0)
        smoothed_score = np.mean(self.score_history)

        return {
            "score": round(smoothed_score, 2),
            "left_ratio": round(left_ratio, 3),
            "right_ratio": round(right_ratio, 3),
            "eye_contact_detected": smoothed_score > 50
        }

    def release(self):
        self.face_mesh.close()


if __name__ == "__main__":
    # Quick test: open webcam and show live eye contact score
    detector = EyeContactDetector()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        result = detector.process_frame(frame)

        score = result.get("score", 0)
        color = (0, 255, 0) if score > 60 else (0, 165, 255) if score > 30 else (0, 0, 255)

        cv2.putText(frame, f"Eye Contact Score: {score:.1f}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(frame, f"Contact: {'YES' if result.get('eye_contact_detected') else 'NO'}",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Eye Contact Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    detector.release()
    cv2.destroyAllWindows()
