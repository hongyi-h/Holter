"""Day encoder: 12-layer Transformer over ~1,563 episode tokens → day-level representation.

Uses standard multi-head self-attention with full global context. At 1,563 tokens
the attention matrix is ~2.4M elements — trivial for modern GPUs. Global attention
directly supports the paper's core claim that 24h temporal relationships matter.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class DayTransformerLayer(nn.Module):
    def __init__(self, d_model: int = 512, n_heads: int = 8, mlp_ratio: int = 4, dropout: float = 0.1):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).reshape(B, T, D)
        x = x + self.out_proj(attn)
        x = x + self.mlp(self.norm2(x))
        return x


class DayEncoder(nn.Module):
    """12-layer Transformer over fused episode tokens.

    Input: episode_waveform (B, n_ep, 384) + episode_rhythm (B, n_ep, 128)
    Output: day embedding (B, 512) + contextualized episode states (B, n_ep, 512)
    """

    def __init__(
        self,
        episode_waveform_dim: int = 384,
        episode_rhythm_dim: int = 128,
        d_model: int = 512,
        n_layers: int = 12,
        n_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        max_episodes: int = 2048,
        n_summary_tokens: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_summary_tokens = n_summary_tokens

        # fuse waveform + rhythm episode tokens
        self.fusion = nn.Sequential(
            nn.Linear(episode_waveform_dim + episode_rhythm_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # learned mask token for masked episode modeling
        self.mask_token = nn.Parameter(torch.randn(d_model) * 0.02)

        # learned summary tokens (like CLS)
        self.summary_tokens = nn.Parameter(torch.randn(1, n_summary_tokens, d_model) * 0.02)

        # sinusoidal position encoding for temporal structure
        self.pos_embed = nn.Parameter(torch.zeros(1, max_episodes + n_summary_tokens, d_model))
        self._init_pos_embed(max_episodes + n_summary_tokens, d_model)

        self.layers = nn.ModuleList([
            DayTransformerLayer(d_model, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # attentive pooling for day embedding
        self.day_attn = nn.Linear(d_model, 1)

    def _init_pos_embed(self, max_len: int, d_model: int):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pos_embed.data.copy_(pe.unsqueeze(0))

    def forward(
        self,
        episode_waveform: torch.Tensor,
        episode_rhythm: torch.Tensor,
        ep_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            episode_waveform: (B, n_ep, 384) from episode encoder CLS tokens
            episode_rhythm: (B, n_ep, 128) from rhythm branch episode pooling
            ep_mask: (B, n_ep) bool, True = masked. If provided, masked positions
                     are replaced before the transformer (80% mask token, 10% random, 10% keep).

        Returns:
            day_embed: (B, 512)
            episode_ctx: (B, n_ep, 512)
            _fused_input: (B, n_ep, 512) detached, pre-mask targets for DayMaskLoss
        """
        B, n_ep, _ = episode_waveform.shape

        z = self.fusion(torch.cat([episode_waveform, episode_rhythm], dim=-1))

        # save pre-mask fused tokens as reconstruction targets
        fused_targets = z.detach()

        # apply BERT-style masking before the transformer sees the tokens
        if ep_mask is not None and ep_mask.any():
            z_masked = z.clone()
            n_masked = ep_mask.sum().item()
            rand_vals = torch.rand(n_masked, device=z.device)
            mask_token_sel = rand_vals < 0.8
            random_sel = (rand_vals >= 0.8) & (rand_vals < 0.9)

            # cast mask_token to match z dtype (bf16 under autocast)
            mt = self.mask_token.to(z.dtype).unsqueeze(0).expand(n_masked, -1)
            z_masked[ep_mask] = torch.where(
                mask_token_sel.unsqueeze(-1).expand(-1, z.shape[-1]),
                mt,
                z_masked[ep_mask],
            )
            if random_sel.any():
                n_random = random_sel.sum().item()
                rand_b = torch.randint(0, B, (n_random,), device=z.device)
                rand_e = torch.randint(0, n_ep, (n_random,), device=z.device)
                random_tokens = z[rand_b, rand_e].detach()
                masked_flat = z_masked[ep_mask]
                masked_flat[random_sel] = random_tokens
                z_masked[ep_mask] = masked_flat
            z = z_masked

        summary = self.summary_tokens.expand(B, -1, -1)
        x = torch.cat([summary, z], dim=1)  # (B, n_summary + n_ep, 512)

        # add positional encoding
        seq_len = x.shape[1]
        x = x + self.pos_embed[:, :seq_len]

        for layer in self.layers:
            x = checkpoint(layer, x, use_reentrant=False)
        x = self.norm(x)

        episode_ctx = x[:, self.n_summary_tokens:]  # (B, n_ep, 512)

        # day embedding: attentive pool over all tokens
        w = self.day_attn(x).squeeze(-1)  # (B, seq_len)
        w = F.softmax(w, dim=-1).unsqueeze(-1)  # (B, seq_len, 1)
        day_embed = (x * w).sum(dim=1)  # (B, 512)

        return {
            "day_embed": day_embed,
            "episode_ctx": episode_ctx,
            "_fused_input": fused_targets,
        }
