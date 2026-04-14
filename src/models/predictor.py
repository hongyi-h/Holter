"""MLP predictor head for SSL."""
import torch.nn as nn


class Predictor(nn.Module):
    """Projects representations for order prediction / downstream."""

    def __init__(self, input_dim=256, hidden_dim=128, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)
