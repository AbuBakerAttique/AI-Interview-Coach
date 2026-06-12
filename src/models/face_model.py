"""
Facial Expression Model — Custom CNN Architecture (From Scratch)
Input: 48x48 grayscale face images
Output: 3 classes — nervous(0), neutral(1), confident(2)
"""

import torch
import torch.nn as nn


class FacialExpressionModel(nn.Module):
    """
    Custom CNN for facial expression classification.
    Built 100% from scratch — no pretrained weights, no VGG, no ResNet.

    Architecture:
        Input → Conv2D Block (x4) → Global Average Pool → Dense → Output
    """

    def __init__(self, num_classes: int = 3, dropout_rate: float = 0.5):
        super(FacialExpressionModel, self).__init__()

        # --- Conv Block 1 ---
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )

        # --- Conv Block 2 ---
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )

        # --- Conv Block 3 ---
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )

        # --- Conv Block 4 ---
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # --- Dense Head ---
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, 1, 48, 48)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


def build_face_model(num_classes: int = 3) -> FacialExpressionModel:
    model = FacialExpressionModel(num_classes=num_classes)
    return model


if __name__ == "__main__":
    model = build_face_model()
    print(model)

    # Test with dummy input
    dummy = torch.randn(8, 1, 48, 48)   # batch=8, channels=1, 48x48
    output = model(dummy)
    print(f"Output shape: {output.shape}")  # Should be (8, 3)
