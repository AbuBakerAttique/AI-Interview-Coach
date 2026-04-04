"""
Train Facial Expression Model
Run: python src/training/train_face.py
"""

import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.models.face_model import build_face_cnn

# --- Config ---
X_TRAIN_PATH = "data/processed/face_features/X_train.npy"
X_VAL_PATH   = "data/processed/face_features/X_val.npy"
Y_TRAIN_PATH = "data/processed/face_features/y_train.npy"
Y_VAL_PATH   = "data/processed/face_features/y_val.npy"
SAVE_PATH    = "saved_models/facial_expression/face_model.keras"
METRICS_PATH = "reports/model_metrics/face_metrics.json"
LOG_DIR      = "reports/training_logs/face"

EPOCHS       = 80
BATCH_SIZE   = 64


def train():
    print("=== Training Facial Expression Model ===\n")

    X_train = np.load(X_TRAIN_PATH)
    X_val   = np.load(X_VAL_PATH)
    y_train = np.load(Y_TRAIN_PATH)
    y_val   = np.load(Y_VAL_PATH)

    print(f"Train: {X_train.shape} | Val: {X_val.shape}")
    print(f"Class distribution (train): {np.bincount(y_train)}\n")

    # Class weights
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}

    # Build model
    model = build_face_cnn(input_shape=X_train.shape[1:], num_classes=3)
    model.summary()

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            SAVE_PATH, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=12, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=6, verbose=1, min_lr=1e-7
        ),
        tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR),
    ]

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

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}")

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
    plt.savefig("reports/model_metrics/face_training_curve.png")
    print("Training curve saved.")


if __name__ == "__main__":
    train()
