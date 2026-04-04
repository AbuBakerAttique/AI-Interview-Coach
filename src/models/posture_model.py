"""
Posture Scorer Model — Neural Network on MediaPipe Landmarks (From Scratch)
Input: 99 features (33 landmarks × [x, y, z])
Output: 5 posture classes
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


POSTURE_CLASSES = ["good_posture", "slouching", "crossed_arms", "leaning_back", "looking_down"]
INPUT_DIM = 99  # 33 landmarks × 3 (x, y, z)


def build_posture_nn(
    input_dim: int = INPUT_DIM,
    num_classes: int = 5,
    dropout_rate: float = 0.4
) -> tf.keras.Model:
    """
    Feedforward Neural Network for posture classification.
    Trained on MediaPipe pose landmark coordinates.
    Built from scratch.

    Args:
        input_dim: Number of input features (99)
        num_classes: 5 posture classes
        dropout_rate: Dropout probability

    Returns:
        Compiled Keras model
    """
    inputs = tf.keras.Input(shape=(input_dim,), name="landmark_input")

    # --- Hidden Layer 1 ---
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    # --- Hidden Layer 2 ---
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    # --- Hidden Layer 3 ---
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    # --- Hidden Layer 4 ---
    x = layers.Dense(32, activation="relu")(x)

    # --- Output ---
    outputs = layers.Dense(num_classes, activation="softmax", name="posture_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="PostureNN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def load_posture_dataset(csv_path: str):
    """
    Load self-collected posture dataset from CSV.
    Returns train/val splits.
    """
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
    model = build_posture_nn()
    model.summary()
