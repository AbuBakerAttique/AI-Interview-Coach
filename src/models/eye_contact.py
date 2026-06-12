"""
Eye Contact Detector — OpenCV Haar Cascade + Iris Geometry
Detects whether the candidate is making eye contact with the camera.
No external dataset needed. Uses eye position geometry.
"""

import cv2
import numpy as np


class EyeContactDetector:
    """
    Detects eye contact using OpenCV face and eye detection.
    Eye contact score: 0 (looking away) → 100 (direct camera contact)
    """

    def __init__(self):
        # Load OpenCV pre-trained detectors
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

        self.score_history = []
        self.history_window = 20  # Smooth over 20 frames

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process a single video frame and return eye contact score.

        Args:
            frame: BGR frame from OpenCV

        Returns:
            dict with keys: score, eye_contact_detected, eyes_found
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect face
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return {"score": 0.0, "eye_contact_detected": False, "eyes_found": 0}

        # Take the largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_roi = gray[y:y+h, x:x+w]

        # Detect eyes inside face region
        eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 5)

        if len(eyes) == 0:
            return {"score": 30.0, "eye_contact_detected": False, "eyes_found": 0}

        frame_center_x = frame.shape[1] / 2
        face_center_x  = x + w / 2

        # How centered is the face horizontally
        deviation = abs(face_center_x - frame_center_x) / (frame.shape[1] / 2)
        face_score = max(0.0, 1.0 - deviation) * 100

        # Check eyes are in upper half of face (means looking forward)
        eyes_in_upper = sum(1 for (ex, ey, ew, eh) in eyes if ey < h * 0.5)
        eye_score = 100.0 if eyes_in_upper >= 1 else 40.0

        # Final score
        score = (face_score * 0.5) + (eye_score * 0.5)

        # Smooth score
        self.score_history.append(score)
        if len(self.score_history) > self.history_window:
            self.score_history.pop(0)
        smoothed = np.mean(self.score_history)

        return {
            "score": round(smoothed, 2),
            "eye_contact_detected": smoothed > 50,
            "eyes_found": len(eyes)
        }

    def release(self):
        pass  # Nothing to release for OpenCV cascade


if __name__ == "__main__":
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

        cv2.putText(frame, f"Eye Contact: {score:.1f}/100", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(frame, f"Eyes found: {result.get('eyes_found', 0)}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Eye Contact Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
