"""1D-CNN beat encoder: single heartbeat -> fixed-dim embedding."""
import torch
import torch.nn as nn


class BeatEncoder(nn.Module):
    """Encode a single heartbeat waveform (beat_samples x channels) -> embed_dim."""

    def __init__(self, in_channels=3, embed_dim=128, beat_samples=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(256, embed_dim)

    def forward(self, x):
        """x: (B, beat_samples, channels) -> (B, embed_dim)"""
        x = x.permute(0, 2, 1)  # (B, C, T)
        x = self.conv(x).squeeze(-1)  # (B, 256)
        return self.fc(x)  # (B, embed_dim)
