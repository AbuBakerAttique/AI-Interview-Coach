"""
Pose Collector — Self-Generated Posture Dataset
Records and labels your own posture dataset using MediaPipe Pose.
Run this script to collect training data for the posture model.
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

POSTURE_LABELS = {
    "1": "good_posture",
    "2": "slouching",
    "3": "crossed_arms",
    "4": "leaning_back",
    "5": "looking_down",
}

LANDMARK_NAMES = [lm.name for lm in mp_pose.PoseLandmark]


def extract_landmarks(results) -> np.ndarray | None:
    """
    Extract 33 pose landmarks as flat feature vector.
    Returns array of shape (99,) → [x, y, z] for each of 33 landmarks
    """
    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks.landmark
    features = []
    for lm in landmarks:
        features.extend([lm.x, lm.y, lm.z])

    return np.array(features)


def collect_posture_data(
    output_path: str = "data/collected/posture_dataset.csv",
    samples_per_class: int = 300,
    collection_delay: float = 0.1
):
    """
    Launch webcam to collect labeled posture samples.

    Controls:
        Press 1-5 to select posture label
        Press SPACE to start/stop recording
        Press Q to quit and save

    Args:
        output_path: Where to save the labeled CSV
        samples_per_class: Target samples per posture class
        collection_delay: Seconds between captured samples
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cap = cv2.VideoCapture(0)
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    collected_data = []
    current_label = None
    recording = False
    last_capture_time = 0
    class_counts = {label: 0 for label in POSTURE_LABELS.values()}

    print("\n=== Posture Data Collector ===")
    print("Press 1-5 to select a posture label:")
    for key, label in POSTURE_LABELS.items():
        print(f"  {key}: {label}")
    print("Press SPACE to start/stop recording")
    print("Press Q to quit and save\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # Draw pose skeleton
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Overlay UI info
        label_text = current_label if current_label else "None selected"
        rec_text = "RECORDING" if recording else "PAUSED"
        rec_color = (0, 0, 255) if recording else (0, 255, 0)

        cv2.putText(frame, f"Label: {label_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, rec_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, rec_color, 2)

        y_offset = 100
        for label, count in class_counts.items():
            cv2.putText(frame, f"{label}: {count}/{samples_per_class}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 22

        cv2.imshow("Posture Data Collector", frame)

        # Capture frame if recording
        current_time = time.time()
        if recording and current_label and results.pose_landmarks:
            if current_time - last_capture_time >= collection_delay:
                landmarks = extract_landmarks(results)
                if landmarks is not None:
                    row = list(landmarks) + [current_label]
                    collected_data.append(row)
                    class_counts[current_label] += 1
                    last_capture_time = current_time

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            recording = not recording
            print(f"Recording {'started' if recording else 'paused'}")
        elif chr(key) in POSTURE_LABELS:
            current_label = POSTURE_LABELS[chr(key)]
            print(f"Label set to: {current_label}")

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

    # Save dataset
    if collected_data:
        feature_cols = [f"{name}_{axis}" for name in LANDMARK_NAMES for axis in ["x", "y", "z"]]
        columns = feature_cols + ["label"]
        df = pd.DataFrame(collected_data, columns=columns)
        df.to_csv(output_path, index=False)
        print(f"\nSaved {len(df)} samples to {output_path}")
        print("Class distribution:")
        print(df["label"].value_counts())
    else:
        print("No data collected.")


if __name__ == "__main__":
    collect_posture_data(
        output_path="data/collected/posture_dataset.csv",
        samples_per_class=300
    )
