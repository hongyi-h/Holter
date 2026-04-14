"""Day-level encoder: sequence of window embeddings -> day embedding."""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DayEncoder(nn.Module):
    """GRU over window embeddings for ordered/shuffled modes."""

    def __init__(self, window_dim=256, hidden_dim=256, n_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=window_dim, hidden_size=hidden_dim,
            num_layers=n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0.0
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, window_embeds, window_mask):
        """
        window_embeds: (B, W, window_dim)
        window_mask: (B, W) bool
        Returns: (day_embed, window_outputs)
        """
        lengths = window_mask.sum(dim=1).clamp(min=1).cpu()
        packed = pack_padded_sequence(window_embeds, lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.gru(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)  # (B, W, hidden_dim)
        day_embed = self.norm(hidden[-1])  # (B, hidden_dim)
        return day_embed, output


class SnippetAggregator(nn.Module):
    """Masked mean pooling for snippet mode (no temporal order)."""

    def __init__(self, window_dim=256):
        super().__init__()
        self.norm = nn.LayerNorm(window_dim)

    def forward(self, window_embeds, window_mask):
        """
        window_embeds: (B, W, window_dim)
        window_mask: (B, W) bool
        Returns: (pooled, window_embeds)
        """
        mask_f = window_mask.unsqueeze(-1).float()  # (B, W, 1)
        summed = (window_embeds * mask_f).sum(dim=1)  # (B, dim)
        count = mask_f.sum(dim=1).clamp(min=1.0)  # (B, 1)
        pooled = self.norm(summed / count)
        return pooled, window_embeds
