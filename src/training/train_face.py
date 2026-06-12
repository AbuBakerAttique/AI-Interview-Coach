"""
Train Facial Expression Model
Run: python src/training/train_face.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
import cv2
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.models.face_model import build_face_model

# --- Config ---
FER_DIR      = "data/raw/fer/images/images/train"
SAVE_PATH    = "saved_models/facial_expression/face_model.pth"
METRICS_PATH = "reports/model_metrics/face_metrics.json"

EPOCHS        = 50
BATCH_SIZE    = 64
LEARNING_RATE = 0.001
IMG_SIZE      = 48

# FER emotion folders → interview label
FER_TO_LABEL = {
    "angry":    0,  # Nervous
    "disgust":  0,  # Nervous
    "fear":     0,  # Nervous
    "sad":      0,  # Nervous
    "neutral":  1,  # Neutral
    "happy":    2,  # Confident
    "surprise": 2,  # Confident
}


def load_fer_dataset(data_dir):
    """Load FER2013 images and preprocess them."""
    images, labels = [], []

    for emotion, label in FER_TO_LABEL.items():
        emotion_dir = os.path.join(data_dir, emotion)
        if not os.path.exists(emotion_dir):
            print(f"Warning: {emotion_dir} not found, skipping.")
            continue

        img_files = [f for f in os.listdir(emotion_dir) if f.endswith(".jpg")]
        print(f"Loading {emotion}: {len(img_files)} images → label {label}")

        for img_file in tqdm(img_files, desc=emotion, leave=False):
            img_path = os.path.join(emotion_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype(np.float32) / 255.0
            images.append(img)
            labels.append(label)

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    # Add channel dimension: (N, 48, 48) → (N, 1, 48, 48)
    images = np.expand_dims(images, axis=1)

    return images, labels


def train():
    print("=== Training Facial Expression Model ===\n")

    # Load dataset
    X, y = load_fer_dataset(FER_DIR)
    print(f"\nLoaded: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {np.bincount(y)} (nervous=0, neutral=1, confident=2)\n")

    # Train/val split manually
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Convert to tensors
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_val_t   = torch.tensor(X_val)
    y_val_t   = torch.tensor(y_val)

    # DataLoaders
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t,   y_val_t),   batch_size=BATCH_SIZE)

    # Build model
    model = build_face_model(num_classes=3)
    print("Model built successfully!\n")

    # Class weights
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_acc = 0.0
    train_accs, val_accs, train_losses, val_losses = [], [], [], []

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    print("Starting training...\n")
    for epoch in range(EPOCHS):

        # --- Train ---
        model.train()
        total_loss, correct, total = 0, 0, 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

        train_loss = total_loss / len(train_loader)
        train_acc  = correct / total

        # --- Validate ---
        model.eval()
        val_loss_total, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss_total += loss.item()
                preds = outputs.argmax(dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total += len(y_batch)

        val_loss = val_loss_total / len(val_loader)
        val_acc  = val_correct / val_total

        scheduler.step(val_loss)

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": 3,
                "img_size": IMG_SIZE,
            }, SAVE_PATH)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {SAVE_PATH}")

    # Save metrics
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    metrics = {
        "best_val_accuracy": float(best_val_acc),
        "epochs_trained": EPOCHS,
        "classes": ["nervous", "neutral", "confident"],
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_accs, label="Train")
    axes[0].plot(val_accs,   label="Val")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(train_losses, label="Train")
    axes[1].plot(val_losses,   label="Val")
    axes[1].set_title("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("reports/model_metrics/face_training_curve.png")
    print("Training curve saved!")


if __name__ == "__main__":
    train()
