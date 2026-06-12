"""
Speech Confidence Model — CNN + LSTM Architecture (From Scratch)
Input: MFCC features (200, 40)
Output: 3 classes — nervous(0), neutral(1), confident(2)
"""

import torch
import torch.nn as nn


class SpeechConfidenceModel(nn.Module):
    """
    Custom CNN + LSTM for speech confidence classification.
    Built 100% from scratch — no pretrained weights.

    Architecture:
        Input → Conv1D (x3) → LSTM → Dense → Output
    """

    def __init__(self, num_classes: int = 3, dropout_rate: float = 0.4):
        super(SpeechConfidenceModel, self).__init__()

        # --- CNN Block 1 ---
        self.conv1 = nn.Sequential(
            nn.Conv1d(40, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate)
        )

        # --- CNN Block 2 ---
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate)
        )

        # --- CNN Block 3 ---
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate)
        )

        # --- LSTM Block ---
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate
        )

        # --- Dense Head ---
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, time_steps, n_mfcc) = (batch, 200, 40)
        x = x.permute(0, 2, 1)   # → (batch, 40, 200) for Conv1d

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.permute(0, 2, 1)   # → (batch, time, channels) for LSTM

        x, _ = self.lstm(x)
        x = x[:, -1, :]          # Take last time step

        x = self.classifier(x)
        return x


def build_speech_model(num_classes: int = 3) -> SpeechConfidenceModel:
    model = SpeechConfidenceModel(num_classes=num_classes)
    return model


if __name__ == "__main__":
    model = build_speech_model()
    print(model)

    # Test with dummy input
    dummy = torch.randn(8, 200, 40)   # batch=8, time=200, mfcc=40
    output = model(dummy)
    print(f"Output shape: {output.shape}")  # Should be (8, 3)
