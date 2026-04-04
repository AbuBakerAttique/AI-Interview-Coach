"""
Train Posture Scorer Model
Run: python src/training/train_posture.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.models.posture_model import build_posture_nn, load_posture_dataset

# --- Config ---
CSV_PATH     = "data/collected/posture_dataset.csv"
SAVE_PATH    = "saved_models/posture_scorer/posture_model.keras"
METRICS_PATH = "reports/model_metrics/posture_metrics.json"
LOG_DIR      = "reports/training_logs/posture"

EPOCHS       = 100
BATCH_SIZE   = 32


def train():
    print("=== Training Posture Scorer Model ===\n")

    X_train, X_val, y_train, y_val, label_encoder = load_posture_dataset(CSV_PATH)

    model = build_posture_nn(input_dim=X_train.shape[1], num_classes=len(label_encoder.classes_))
    model.summary()

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            SAVE_PATH, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=15, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7, verbose=1
        ),
        tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR),
    ]

    print("\nStarting training...\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}")

    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    metrics = {
        "val_accuracy": float(val_acc),
        "val_loss": float(val_loss),
        "classes": list(label_encoder.classes_),
        "best_val_accuracy": float(max(history.history["val_accuracy"])),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title("Posture Accuracy")
    axes[0].legend()
    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title("Loss")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("reports/model_metrics/posture_training_curve.png")
    print("Training curve saved.")


if __name__ == "__main__":
    train()
