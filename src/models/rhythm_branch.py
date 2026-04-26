"""Rhythm branch: bidirectional Mamba over beat-level rhythm tokens (~100k/day)."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class MambaBlock(nn.Module):
    """Simplified Mamba-style SSM block with local conv + selective gating."""

    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, expand: int = 2):
        super().__init__()
        d_inner = d_model * expand
        self.d_inner = d_inner
        self.d_state = d_state

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # local conv
        self.conv = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv - 1, groups=d_inner)

        # SSM parameters
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float().unsqueeze(0).expand(d_inner, -1)).contiguous())
        self.D = nn.Parameter(torch.ones(d_inner))
        self.B_proj = nn.Linear(d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(d_inner, d_state, bias=False)

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = self.norm(x)
        xz = self.in_proj(h)
        x_ssm, z = xz.chunk(2, dim=-1)

        # local conv
        x_ssm = x_ssm.transpose(1, 2)
        x_ssm = self.conv(x_ssm)[:, :, :T]
        x_ssm = x_ssm.transpose(1, 2)
        x_ssm = F.silu(x_ssm)

        # selective SSM (simplified sequential scan)
        dt = F.softplus(self.dt_proj(x_ssm))  # (B, T, d_inner)
        A = -torch.exp(self.A_log)             # (d_inner, d_state)
        B_t = self.B_proj(x_ssm)              # (B, T, d_state)
        C_t = self.C_proj(x_ssm)              # (B, T, d_state)

        # parallel-friendly: use chunked scan for long sequences
        y = self._scan(x_ssm, dt, A, B_t, C_t)
        y = y + x_ssm * self.D.unsqueeze(0).unsqueeze(0)

        y = y * F.silu(z)
        return x + self.out_proj(y)

    def _scan(self, x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor,
              B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        B_sz, T, d_inner = x.shape
        d_state = self.d_state

        chunk_size = min(T, 2048)
        if T <= chunk_size:
            return self._scan_chunk(x, dt, A, B, C)

        outputs = []
        state = torch.zeros(B_sz, d_inner, d_state, device=x.device, dtype=x.dtype)
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            y_chunk, state = checkpoint(
                self._scan_chunk_stateful,
                x[:, start:end], dt[:, start:end],
                A, B[:, start:end], C[:, start:end], state,
                use_reentrant=False,
            )
            outputs.append(y_chunk)
        return torch.cat(outputs, dim=1)

    def _scan_chunk(self, x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor,
                    B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        B_sz, T, d_inner = x.shape
        state = torch.zeros(B_sz, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(T):
            dA = torch.exp(dt[:, t].unsqueeze(-1) * A.unsqueeze(0))  # (B, d_inner, d_state)
            dB = dt[:, t].unsqueeze(-1) * B[:, t].unsqueeze(1)       # (B, d_inner, d_state)
            state = state * dA + dB * x[:, t].unsqueeze(-1)
            y_t = (state * C[:, t].unsqueeze(1)).sum(dim=-1)          # (B, d_inner)
            outputs.append(y_t)
        return torch.stack(outputs, dim=1)

    def _scan_chunk_stateful(self, x, dt, A, B, C, state):
        B_sz, T, d_inner = x.shape
        outputs = []
        for t in range(T):
            dA = torch.exp(dt[:, t].unsqueeze(-1) * A.unsqueeze(0))
            dB = dt[:, t].unsqueeze(-1) * B[:, t].unsqueeze(1)
            state = state * dA + dB * x[:, t].unsqueeze(-1)
            y_t = (state * C[:, t].unsqueeze(1)).sum(dim=-1)
            outputs.append(y_t)
        return torch.stack(outputs, dim=1), state


class BiMamba(nn.Module):
    """Bidirectional Mamba: forward + backward SSM."""

    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.fwd = MambaBlock(d_model, d_state, d_conv, expand)
        self.bwd = MambaBlock(d_model, d_state, d_conv, expand)
        self.gate = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_fwd = self.fwd(x)
        h_bwd = self.bwd(x.flip(1)).flip(1)
        return self.gate(torch.cat([h_fwd, h_bwd], dim=-1))


class RhythmBranch(nn.Module):
    """8-layer BiMamba over rhythm tokens (VQ code + RR bins + clock features).

    Processes ~100k beat-level tokens per day.
    ~4.6M parameters.
    """

    def __init__(
        self,
        n_codes: int = 512,
        n_rr_bins: int = 32,
        d_model: int = 128,
        d_state: int = 64,
        n_layers: int = 8,
        d_conv: int = 4,
        episode_len: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.episode_len = episode_len

        # token embeddings
        self.code_embed = nn.Embedding(n_codes, 64)
        self.rr_embed = nn.Embedding(n_rr_bins, 16)
        self.rr_prev_embed = nn.Embedding(n_rr_bins, 16)
        self.clock_proj = nn.Linear(2, 8)  # sin, cos

        # project to d_model
        self.input_proj = nn.Linear(64 + 16 + 16 + 8, d_model)

        self.layers = nn.ModuleList([
            BiMamba(d_model, d_state, d_conv, expand=1) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

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
            code_idx: (B, T) VQ code indices
            rr_bins: (B, T) quantized RR interval bins
            rr_prev_bins: (B, T) quantized previous RR bins
            hour_sin, hour_cos: (B, T) clock features

        Returns:
            beat_rhythm: (B, T, d_model) per-beat rhythm states
            episode_rhythm: (B, n_ep, d_model) mean-pooled per episode
        """
        h_code = self.code_embed(code_idx)
        h_rr = self.rr_embed(rr_bins)
        h_rr_prev = self.rr_prev_embed(rr_prev_bins)
        h_clock = self.clock_proj(torch.stack([hour_sin, hour_cos], dim=-1))

        x = self.input_proj(torch.cat([h_code, h_rr, h_rr_prev, h_clock], dim=-1))

        for layer in self.layers:
            x = checkpoint(layer, x, use_reentrant=False)
        x = self.norm(x)

        # episode-level pooling
        B, T, D = x.shape
        n_ep = T // self.episode_len
        ep_len = n_ep * self.episode_len
        ep_rhythm = x[:, :ep_len].reshape(B, n_ep, self.episode_len, D).mean(dim=2)

        return {
            "beat_rhythm": x,        # (B, T, 128)
            "episode_rhythm": ep_rhythm,  # (B, n_ep, 128)
        }
