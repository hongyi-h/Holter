"""Rhythm branch: episode-local processing of beat-level rhythm tokens.

Each 64-beat episode is processed independently — no cross-episode sequence modeling.
This is physiologically justified: rhythm context (coupling intervals, short-term HRV,
ectopy patterns) operates at the ±5 beat to 5-minute scale, all within one episode.

Circadian context is provided by hour_sin/hour_cos input features and by the
DayEncoder's global attention over episode tokens.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RhythmBlock(nn.Module):
    """Pre-norm feed-forward block with 1D local convolution for rhythm tokens."""

    def __init__(self, d_model: int = 128, kernel_size: int = 7, mlp_ratio: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=kernel_size // 2, groups=d_model,
        )
        self.proj = nn.Linear(d_model, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        h = self.norm1(x)
        h = h.transpose(1, 2)  # (B, D, T)
        h = self.conv(h).transpose(1, 2)  # (B, T, D)
        x = x + self.proj(F.gelu(h))
        x = x + self.mlp(self.norm2(x))
        return x


class RhythmBranch(nn.Module):
    """Episode-local rhythm encoder: processes each 64-beat episode independently.

    Input: rhythm tokens (VQ code + RR bins + clock features) for the full day
    Processing: reshape into episodes, encode each independently with local conv + MLP
    Output: per-beat rhythm states + per-episode rhythm summary

    ~0.8M parameters.
    """

    def __init__(
        self,
        n_codes: int = 512,
        n_rr_bins: int = 32,
        d_model: int = 128,
        n_layers: int = 4,
        kernel_size: int = 7,
        episode_len: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.episode_len = episode_len

        # token embeddings
        self.code_embed = nn.Embedding(n_codes, 64)
        self.rr_embed = nn.Embedding(n_rr_bins, 16)
        self.rr_prev_embed = nn.Embedding(n_rr_bins, 16)
        self.clock_proj = nn.Linear(2, 8)

        # project to d_model
        self.input_proj = nn.Linear(64 + 16 + 16 + 8, d_model)

        # episode-local layers (each episode processed independently)
        self.layers = nn.ModuleList([
            RhythmBlock(d_model, kernel_size) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # episode summary: attentive pooling over 64 beats
        self.ep_attn = nn.Linear(d_model, 1)

    def forward(
        self,
        code_idx: torch.Tensor,
        rr_bins: torch.Tensor,
        rr_prev_bins: torch.Tensor,
        hour_sin: torch.Tensor,
        hour_cos: torch.Tensor,
        n_beats: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            code_idx: (B, max_beats) VQ code indices
            rr_bins: (B, max_beats) quantized RR interval
            rr_prev_bins: (B, max_beats) quantized previous RR
            hour_sin/cos: (B, max_beats) clock features

        Returns:
            beat_rhythm: (B, max_beats, 128) per-beat rhythm states
            episode_rhythm: (B, n_ep, 128) per-episode rhythm summaries
        """
        B, T = code_idx.shape

        # embed rhythm tokens
        h_code = self.code_embed(code_idx)
        h_rr = self.rr_embed(rr_bins)
        h_rr_prev = self.rr_prev_embed(rr_prev_bins)
        h_clock = self.clock_proj(torch.stack([hour_sin, hour_cos], dim=-1))

        x = self.input_proj(torch.cat([h_code, h_rr, h_rr_prev, h_clock], dim=-1))
        # x: (B, T, d_model)

        # reshape into episodes for local processing
        ep_len = self.episode_len
        n_ep = T // ep_len
        ep_beats = n_ep * ep_len

        if n_ep > 0:
            # (B, n_ep, ep_len, d_model)
            x_ep = x[:, :ep_beats].reshape(B * n_ep, ep_len, self.d_model)

            for layer in self.layers:
                x_ep = layer(x_ep)
            x_ep = self.norm(x_ep)

            # reshape back
            x_ep = x_ep.reshape(B, n_ep, ep_len, self.d_model)

            # write back to full sequence
            x_out = x.clone()
            x_out[:, :ep_beats] = x_ep.reshape(B, ep_beats, self.d_model)

            # episode summary via attentive pooling
            w = self.ep_attn(x_ep).squeeze(-1)  # (B, n_ep, ep_len)
            w = F.softmax(w, dim=-1).unsqueeze(-1)  # (B, n_ep, ep_len, 1)
            ep_rhythm = (x_ep * w).sum(dim=2)  # (B, n_ep, d_model)
        else:
            x_out = x
            ep_rhythm = torch.zeros(B, 0, self.d_model, device=x.device)

        return {
            "beat_rhythm": x_out,       # (B, T, 128)
            "episode_rhythm": ep_rhythm, # (B, n_ep, 128)
        }
