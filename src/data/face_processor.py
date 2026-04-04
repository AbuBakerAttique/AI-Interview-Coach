"""
Face Processor — Facial Expression Model
Preprocesses FER2013 dataset images for CNN training.
"""

import os
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# FER2013 class names → interview-relevant remapping
FER_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

FER_TO_INTERVIEW = {
    "angry":    "nervous",
    "disgust":  "nervous",
    "fear":     "nervous",
    "happy":    "confident",
    "neutral":  "neutral",
    "sad":      "nervous",
    "surprise": "confident",
}

LABEL_ENCODING = {"nervous": 0, "neutral": 1, "confident": 2}

IMG_SIZE = (48, 48)


def preprocess_image(img_path: str) -> np.ndarray:
    """
    Load and preprocess a face image.
    Returns normalized grayscale array of shape (48, 48, 1)
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=-1)    # Shape: (48, 48, 1)
    return img


def augment_image(img: np.ndarray) -> list:
    """
    Apply augmentation: flip, rotate, brightness shift.
    """
    augmented = []
    img_squeezed = img[:, :, 0]  # Remove channel dim for OpenCV

    # Horizontal flip
    flipped = cv2.flip(img_squeezed, 1)
    augmented.append(np.expand_dims(flipped, axis=-1))

    # Rotate ±10 degrees
    h, w = img_squeezed.shape
    for angle in [-10, 10]:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(img_squeezed, M, (w, h))
        augmented.append(np.expand_dims(rotated, axis=-1))

    # Brightness shift
    bright = np.clip(img_squeezed + 0.1, 0, 1)
    augmented.append(np.expand_dims(bright, axis=-1))

    return augmented


def process_fer_dataset(
    data_dir: str,
    output_dir: str,
    augment: bool = True,
    test_size: float = 0.2
):
    """
    Process FER2013 dataset from folder structure.

    Expected structure:
        data/raw/fer/
            train/
                angry/ *.jpg
                happy/ *.jpg
                ...
            validation/ (or test/)
                angry/ *.jpg
                ...

    Args:
        data_dir: Path to FER2013 root folder
        output_dir: Where to save processed .npy files
        augment: Apply augmentation to training set
        test_size: Validation split ratio
    """
    os.makedirs(output_dir, exist_ok=True)

    images = []
    labels = []

    train_dir = os.path.join(data_dir, "train")

    for class_name in FER_CLASSES:
        class_dir = os.path.join(train_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} not found, skipping.")
            continue

        interview_label = FER_TO_INTERVIEW[class_name]
        encoded = LABEL_ENCODING[interview_label]

        img_files = [f for f in os.listdir(class_dir) if f.endswith((".jpg", ".png"))]
        print(f"Processing {class_name}: {len(img_files)} images → {interview_label}")

        for img_file in tqdm(img_files, desc=class_name, leave=False):
            img_path = os.path.join(class_dir, img_file)
            img = preprocess_image(img_path)
            if img is None:
                continue

            images.append(img)
            labels.append(encoded)

            if augment:
                for aug_img in augment_image(img):
                    images.append(aug_img)
                    labels.append(encoded)

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    # Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=test_size, random_state=42, stratify=labels
    )

    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val)

    print(f"\nSaved to {output_dir}")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Class distribution (train): {np.bincount(y_train)} (nervous, neutral, confident)")


if __name__ == "__main__":
    process_fer_dataset(
        data_dir="data/raw/fer",
        output_dir="data/processed/face_features",
        augment=True
    )
