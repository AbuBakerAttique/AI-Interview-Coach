"""
Train Speech Confidence Model
Run: python src/training/train_speech.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import librosa
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.models.speech_model import build_speech_model

# --- Config ---
RAVDESS_DIR  = "data/raw/ravdess"
SAVE_PATH    = "saved_models/speech_confidence/speech_model.pth"
METRICS_PATH = "reports/model_metrics/speech_metrics.json"

EPOCHS        = 60
BATCH_SIZE    = 32
LEARNING_RATE = 0.001
N_MFCC        = 40
MAX_LEN       = 200

# RAVDESS emotion → interview label
EMOTION_MAP = {
    "01": 1,  # neutral  → Neutral
    "02": 1,  # calm     → Neutral
    "03": 2,  # happy    → Confident
    "04": 0,  # sad      → Nervous
    "05": 0,  # angry    → Nervous
    "06": 0,  # fearful  → Nervous
    "07": 2,  # disgust  → Confident
    "08": 2,  # surprised→ Confident
}


def extract_mfcc(file_path):
    """Extract MFCC features from a .wav file."""
    audio, sr = librosa.load(file_path, res_type="kaiser_fast")
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC).T

    if mfcc.shape[0] < MAX_LEN:
        mfcc = np.pad(mfcc, ((0, MAX_LEN - mfcc.shape[0]), (0, 0)))
    else:
        mfcc = mfcc[:MAX_LEN, :]

    return mfcc


def load_ravdess(data_dir):
    """Load all RAVDESS audio files and extract MFCC features."""
    features, labels = [], []

    wav_files = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".wav"):
                wav_files.append(os.path.join(root, f))

    print(f"Found {len(wav_files)} audio files. Extracting MFCC features...")

    for file_path in tqdm(wav_files):
        filename = os.path.basename(file_path)
        parts = filename.split("-")
        emotion_code = parts[2]
        label = EMOTION_MAP.get(emotion_code, 1)

        mfcc = extract_mfcc(file_path)
        features.append(mfcc)
        labels.append(label)

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64)


def train():
    print("=== Training Speech Confidence Model ===\n")

    # Load and extract features
    X, y = load_ravdess(RAVDESS_DIR)
    print(f"\nLoaded: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {np.bincount(y)} (nervous=0, neutral=1, confident=2)\n")

    # Train/val split
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
    model = build_speech_model(num_classes=3)
    print(f"Model built successfully!\n")

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
                "n_mfcc": N_MFCC,
                "max_len": MAX_LEN,
            }, SAVE_PATH)

        if (epoch + 1) % 10 == 0:
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
    plt.savefig("reports/model_metrics/speech_training_curve.png")
    print("Training curve saved!")


if __name__ == "__main__":
    train()
