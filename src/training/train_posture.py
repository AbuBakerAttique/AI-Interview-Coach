"""
Train Posture Scorer Model
Run: python src/training/train_posture.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.models.posture_model import build_posture_model, load_posture_dataset

# --- Config ---
CSV_PATH     = "data/raw/posture/data.csv"
SAVE_PATH    = "saved_models/posture_scorer/posture_model.pth"
METRICS_PATH = "reports/model_metrics/posture_metrics.json"

EPOCHS       = 100
BATCH_SIZE   = 32
LEARNING_RATE = 0.001


def train():
    print("=== Training Posture Scorer Model ===\n")

    # Load dataset
    X_train, X_val, y_train, y_val, label_encoder = load_posture_dataset(CSV_PATH)

    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
    y_val_t   = torch.tensor(y_val,   dtype=torch.long)

    # DataLoaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset   = TensorDataset(X_val_t,   y_val_t)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE)

    # Build model
    num_classes = len(label_encoder.classes_)
    model = build_posture_model(input_dim=X_train.shape[1], num_classes=num_classes)
    print(f"\nModel built. Input dim: {X_train.shape[1]}, Classes: {label_encoder.classes_}\n")

    # Class weights to handle imbalance
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.5)

    # Training loop
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
                "label_encoder_classes": label_encoder.classes_,
                "input_dim": X_train.shape[1],
                "num_classes": num_classes,
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
        "classes": list(label_encoder.classes_),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_accs, label="Train")
    axes[0].plot(val_accs,   label="Val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(train_losses, label="Train")
    axes[1].plot(val_losses,   label="Val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    os.makedirs("reports/model_metrics", exist_ok=True)
    plt.savefig("reports/model_metrics/posture_training_curve.png")
    print("Training curve saved to reports/model_metrics/posture_training_curve.png")


if __name__ == "__main__":
    train()
