"""
Posture Scorer Model — Neural Network on MediaPipe Landmarks (From Scratch)
Input: 99 features (33 landmarks x [x, y, z])
Output: 5 posture classes
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


POSTURE_CLASSES = ["good_posture", "slouching", "crossed_arms", "leaning_back", "looking_down"]
INPUT_DIM = 99  # 33 landmarks x 3 (x, y, z)


class PostureModel(nn.Module):
    """
    Feedforward Neural Network for posture classification.
    Trained on MediaPipe pose landmark coordinates.
    Built from scratch.
    """

    def __init__(self, input_dim: int = INPUT_DIM, num_classes: int = 5, dropout_rate: float = 0.4):
        super(PostureModel, self).__init__()

        self.network = nn.Sequential(
            # --- Hidden Layer 1 ---
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # --- Hidden Layer 2 ---
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # --- Hidden Layer 3 ---
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # --- Hidden Layer 4 ---
            nn.Linear(64, 32),
            nn.ReLU(),

            # --- Output ---
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)


def build_posture_model(input_dim: int = INPUT_DIM, num_classes: int = 5) -> PostureModel:
    model = PostureModel(input_dim=input_dim, num_classes=num_classes)
    return model


def load_posture_dataset(csv_path: str):
    """Load self-collected posture dataset from CSV."""
    df = pd.read_csv(csv_path)

    X = df.drop("label", axis=1).values.astype(np.float32)
    y_raw = df["label"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Classes: {le.classes_}")

    return X_train, X_val, y_train, y_val, le


if __name__ == "__main__":
    model = build_posture_model()
    print(model)

    # Test with dummy input
    dummy = torch.randn(8, 99)   # batch=8, 99 landmark features
    output = model(dummy)
    print(f"Output shape: {output.shape}")  # Should be (8, 5)
