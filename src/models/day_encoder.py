"""Day encoder: 12-layer BiMamba over ~1,563 episode tokens → day-level representation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rhythm_branch import BiMamba


class DayEncoder(nn.Module):
    """12-layer BiMamba over fused episode tokens.

    Input: (B, n_episodes, 512) fused episode tokens
    Output: day embedding (B, 512) + contextualized episode states (B, n_episodes, 512)
    ~24.8M parameters (including fusion MLPs and task projections).
    """

    def __init__(
        self,
        episode_waveform_dim: int = 384,
        episode_rhythm_dim: int = 128,
        d_model: int = 512,
        d_state: int = 64,
        n_layers: int = 12,
        d_conv: int = 4,
        n_summary_tokens: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_summary_tokens = n_summary_tokens

        self.fusion = nn.Sequential(
            nn.Linear(episode_waveform_dim + episode_rhythm_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.summary_tokens = nn.Parameter(torch.randn(1, n_summary_tokens, d_model) * 0.02)

        self.layers = nn.ModuleList([
            BiMamba(d_model, d_state, d_conv, expand=1) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        self.day_attn = nn.Linear(d_model, 1)

    def forward(
        self,
        episode_waveform: torch.Tensor,
        episode_rhythm: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            episode_waveform: (B, n_ep, 384) from episode encoder CLS tokens
            episode_rhythm: (B, n_ep, 128) from rhythm branch episode pooling

        Returns:
            day_embed: (B, 512)
            episode_ctx: (B, n_ep, 512)
        """
        B, n_ep, _ = episode_waveform.shape

        z = self.fusion(torch.cat([episode_waveform, episode_rhythm], dim=-1))

        summary = self.summary_tokens.expand(B, -1, -1)
        x = torch.cat([summary, z], dim=1)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        episode_ctx = x[:, self.n_summary_tokens:]

        w = self.day_attn(x).squeeze(-1)
        w = F.softmax(w, dim=-1).unsqueeze(-1)
        day_embed = (x * w).sum(dim=1)

        return {
            "day_embed": day_embed,
            "episode_ctx": episode_ctx,
            "_fused_input": z.detach(),
        }
