"""
Audio Processor — Speech Confidence Model
Extracts MFCC features from RAVDESS dataset audio files.
"""

import os
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm


# RAVDESS emotion label map
# We remap 8 RAVDESS emotions → 3 interview-relevant classes
RAVDESS_EMOTION_MAP = {
    1: "neutral",    # neutral    → Neutral
    2: "neutral",    # calm       → Neutral
    3: "confident",  # happy      → Confident
    4: "nervous",    # sad        → Nervous
    5: "nervous",    # angry      → Nervous
    6: "nervous",    # fearful    → Nervous
    7: "confident",  # disgust    → Confident (assertive)
    8: "confident",  # surprised  → Confident
}

LABEL_ENCODING = {"nervous": 0, "neutral": 1, "confident": 2}


def extract_mfcc(file_path: str, n_mfcc: int = 40, max_len: int = 200) -> np.ndarray:
    """
    Extract MFCC features from a .wav file.

    Args:
        file_path: Path to .wav audio file
        n_mfcc: Number of MFCC coefficients
        max_len: Max time frames (pad or truncate)

    Returns:
        MFCC feature array of shape (max_len, n_mfcc)
    """
    audio, sample_rate = librosa.load(file_path, res_type="kaiser_fast")

    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    mfcc = mfcc.T  # shape: (time_steps, n_mfcc)

    # Pad or truncate to fixed length
    if mfcc.shape[0] < max_len:
        pad_width = max_len - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode="constant")
    else:
        mfcc = mfcc[:max_len, :]

    return mfcc


def augment_audio(audio: np.ndarray, sample_rate: int) -> list:
    """
    Data augmentation: add noise, pitch shift, time stretch.
    Returns a list of augmented audio arrays.
    """
    augmented = []

    # Add white noise
    noise = np.random.randn(len(audio)) * 0.005
    augmented.append(audio + noise)

    # Pitch shift up
    augmented.append(librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=2))

    # Pitch shift down
    augmented.append(librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=-2))

    # Time stretch
    augmented.append(librosa.effects.time_stretch(audio, rate=0.9))

    return augmented


def parse_ravdess_label(filename: str) -> int:
    """
    Parse emotion label from RAVDESS filename.
    Format: 03-01-06-01-02-01-12.wav
    Position 2 (index) = emotion (1-8)
    """
    parts = filename.split("-")
    emotion_code = int(parts[2])
    return emotion_code


def process_ravdess_dataset(
    data_dir: str,
    output_dir: str,
    n_mfcc: int = 40,
    max_len: int = 200,
    augment: bool = True
):
    """
    Process entire RAVDESS dataset and save features as .npy arrays.

    Args:
        data_dir: Path to raw RAVDESS audio files (data/raw/ravdess/)
        output_dir: Path to save processed features (data/processed/audio_features/)
        n_mfcc: Number of MFCC coefficients
        max_len: Fixed time frame length
        augment: Whether to apply data augmentation
    """
    os.makedirs(output_dir, exist_ok=True)

    features = []
    labels = []

    audio_files = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".wav"):
                audio_files.append(os.path.join(root, f))

    print(f"Found {len(audio_files)} audio files in {data_dir}")

    for file_path in tqdm(audio_files, desc="Extracting MFCC features"):
        filename = os.path.basename(file_path)

        try:
            emotion_code = parse_ravdess_label(filename)
            interview_label = RAVDESS_EMOTION_MAP.get(emotion_code, "neutral")
            encoded_label = LABEL_ENCODING[interview_label]

            # Extract MFCC from original
            mfcc = extract_mfcc(file_path, n_mfcc=n_mfcc, max_len=max_len)
            features.append(mfcc)
            labels.append(encoded_label)

            # Augment training data
            if augment:
                audio, sr = librosa.load(file_path, res_type="kaiser_fast")
                augmented_audios = augment_audio(audio, sr)
                for aug_audio in augmented_audios:
                    aug_mfcc = librosa.feature.mfcc(y=aug_audio, sr=sr, n_mfcc=n_mfcc).T
                    if aug_mfcc.shape[0] < max_len:
                        aug_mfcc = np.pad(aug_mfcc, ((0, max_len - aug_mfcc.shape[0]), (0, 0)))
                    else:
                        aug_mfcc = aug_mfcc[:max_len, :]
                    features.append(aug_mfcc)
                    labels.append(encoded_label)

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    features = np.array(features)
    labels = np.array(labels)

    np.save(os.path.join(output_dir, "mfcc_features.npy"), features)
    np.save(os.path.join(output_dir, "labels.npy"), labels)

    print(f"\nSaved {len(features)} samples to {output_dir}")
    print(f"Feature shape: {features.shape}")
    print(f"Class distribution: {np.bincount(labels)} (nervous, neutral, confident)")


if __name__ == "__main__":
    process_ravdess_dataset(
        data_dir="data/raw/ravdess",
        output_dir="data/processed/audio_features",
        augment=True
    )
