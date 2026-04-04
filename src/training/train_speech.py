"""
Train Speech Confidence Model
Run: python src/training/train_speech.py
"""

import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.models.speech_model import build_speech_cnn_lstm

# --- Config ---
FEATURES_PATH = "data/processed/audio_features/mfcc_features.npy"
LABELS_PATH   = "data/processed/audio_features/labels.npy"
SAVE_PATH     = "saved_models/speech_confidence/speech_model.keras"
METRICS_PATH  = "reports/model_metrics/speech_metrics.json"
LOG_DIR       = "reports/training_logs/speech"

EPOCHS        = 60
BATCH_SIZE    = 32
VAL_SPLIT     = 0.2
RANDOM_SEED   = 42


def train():
    print("=== Training Speech Confidence Model ===\n")

    # Load data
    print("Loading features...")
    X = np.load(FEATURES_PATH)
    y = np.load(LABELS_PATH)
    print(f"Loaded: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {np.bincount(y)} (nervous=0, neutral=1, confident=2)\n")

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SPLIT, random_state=RANDOM_SEED, stratify=y
    )

    # Class weights to handle imbalance
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    print(f"Class weights: {class_weight_dict}\n")

    # Build model
    model = build_speech_cnn_lstm(input_shape=(X.shape[1], X.shape[2]), num_classes=3)
    model.summary()

    # Callbacks
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            SAVE_PATH, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, verbose=1, min_lr=1e-6
        ),
        tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR),
    ]

    # Train
    print("\nStarting training...\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}")

    # Save metrics
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    metrics = {
        "val_accuracy": float(val_acc),
        "val_loss": float(val_loss),
        "epochs_trained": len(history.history["accuracy"]),
        "best_val_accuracy": float(max(history.history["val_accuracy"])),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("reports/model_metrics/speech_training_curve.png")
    print("Training curve saved.")


if __name__ == "__main__":
    train()
