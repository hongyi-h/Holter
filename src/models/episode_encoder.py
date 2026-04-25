"""Episode encoder: 6-layer Transformer over 64 beats within an episode."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int = 256):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_len = max_len

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        return freqs.cos(), freqs.sin()


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D)
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class EpisodeTransformerLayer(nn.Module):
    def __init__(self, d_model: int = 384, n_heads: int = 6, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = apply_rotary(q, rope_cos, rope_sin)
        k = apply_rotary(k, rope_cos, rope_sin)

        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        attn = attn.transpose(1, 2).reshape(B, T, D)
        x = x + self.out_proj(attn)
        x = x + self.mlp(self.norm2(x))
        return x


class EpisodeEncoder(nn.Module):
    """6-layer Transformer encoder over 64 beats within an episode.

    Input: beat embeddings (B*n_ep, 64, beat_dim) → projected to d_model=384
    Output: contextualized beat states (B*n_ep, 64, 384) + episode CLS token (B*n_ep, 384)
    ~11.3M parameters.
    """

    def __init__(
        self,
        beat_dim: int = 256,
        d_model: int = 384,
        n_layers: int = 6,
        n_heads: int = 6,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        episode_len: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.episode_len = episode_len

        self.input_proj = nn.Linear(beat_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.rope = RotaryEmbedding(d_model // n_heads, max_len=episode_len + 1)

        self.layers = nn.ModuleList([
            EpisodeTransformerLayer(d_model, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, beat_embeds: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            beat_embeds: (B, episode_len, beat_dim) — beat embeddings for one episode

        Returns:
            beat_ctx: (B, episode_len, d_model) — contextualized beat states
            episode_token: (B, d_model) — CLS token for the episode
        """
        B = beat_embeds.shape[0]
        x = self.input_proj(beat_embeds)  # (B, 64, 384)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 65, 384)

        cos, sin = self.rope(x.shape[1], x.device)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D/2)
        sin = sin.unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)

        return {
            "beat_ctx": x[:, 1:],    # (B, 64, 384)
            "episode_token": x[:, 0], # (B, 384)
        }
