"""Rhythm branch: bidirectional Mamba over beat-level rhythm tokens (~100k/day).

Uses official mamba-ssm Mamba2 on CUDA, falls back to a pure-PyTorch
minimal SSM on MPS / CPU for local development.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Backend selection: official mamba-ssm (CUDA) vs pure-PyTorch fallback (MPS/CPU)
# ---------------------------------------------------------------------------
try:
    from mamba_ssm.modules.mamba2 import Mamba2 as _OfficialMamba2
    HAS_MAMBA_SSM = True
except ImportError:
    HAS_MAMBA_SSM = False


class _FallbackMamba(nn.Module):
    """Minimal selective-SSM for MPS/CPU development. NOT for training."""

    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, expand: int = 2):
        super().__init__()
        d_inner = d_model * expand
        self.d_inner = d_inner
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv - 1, groups=d_inner)
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1).float().unsqueeze(0).expand(d_inner, -1)).contiguous()
        )
        self.D = nn.Parameter(torch.ones(d_inner))
        self.B_proj = nn.Linear(d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(d_inner, d_state, bias=False)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        x_ssm = x_ssm.transpose(1, 2)
        x_ssm = self.conv(x_ssm)[:, :, :T]
        x_ssm = x_ssm.transpose(1, 2)
        x_ssm = F.silu(x_ssm)

        dt = F.softplus(self.dt_proj(x_ssm))
        A = -torch.exp(self.A_log)
        B_t = self.B_proj(x_ssm)
        C_t = self.C_proj(x_ssm)

        y = self._scan(x_ssm, dt, A, B_t, C_t)
        y = y + x_ssm * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)
        return self.out_proj(y)

    def _scan(self, x, dt, A, B, C):
        B_sz, T, d_inner = x.shape
        state = torch.zeros(B_sz, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(T):
            dA = torch.exp(dt[:, t].unsqueeze(-1) * A.unsqueeze(0))
            dB = dt[:, t].unsqueeze(-1) * B[:, t].unsqueeze(1)
            state = state * dA + dB * x[:, t].unsqueeze(-1)
            y_t = (state * C[:, t].unsqueeze(1)).sum(dim=-1)
            outputs.append(y_t)
        return torch.stack(outputs, dim=1)


def _make_mamba_inner(d_model: int, d_state: int = 64, d_conv: int = 4, expand: int = 2) -> nn.Module:
    """Create a single-direction Mamba layer, choosing backend by availability."""
    if HAS_MAMBA_SSM and torch.cuda.is_available():
        headdim = min(64, d_model * expand)
        return _OfficialMamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        )
    return _FallbackMamba(d_model, d_state, d_conv, expand)


# ---------------------------------------------------------------------------
# MambaBlock: norm + Mamba + residual
# ---------------------------------------------------------------------------
class MambaBlock(nn.Module):
    """Pre-norm Mamba block with residual connection."""

    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = _make_mamba_inner(d_model, d_state, d_conv, expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mamba(self.norm(x))


# ---------------------------------------------------------------------------
# BiMamba: bidirectional wrapper
# ---------------------------------------------------------------------------
class BiMamba(nn.Module):
    """Bidirectional Mamba: forward + backward SSM with gated fusion."""

    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.fwd = MambaBlock(d_model, d_state, d_conv, expand)
        self.bwd = MambaBlock(d_model, d_state, d_conv, expand)
        self.gate = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_fwd = self.fwd(x)
        h_bwd = self.bwd(x.flip(1)).flip(1)
        return self.gate(torch.cat([h_fwd, h_bwd], dim=-1))


# ---------------------------------------------------------------------------
# RhythmBranch
# ---------------------------------------------------------------------------
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

        self.code_embed = nn.Embedding(n_codes, 64)
        self.rr_embed = nn.Embedding(n_rr_bins, 16)
        self.rr_prev_embed = nn.Embedding(n_rr_bins, 16)
        self.clock_proj = nn.Linear(2, 8)

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
        h_code = self.code_embed(code_idx)
        h_rr = self.rr_embed(rr_bins)
        h_rr_prev = self.rr_prev_embed(rr_prev_bins)
        h_clock = self.clock_proj(torch.stack([hour_sin, hour_cos], dim=-1))

        x = self.input_proj(torch.cat([h_code, h_rr, h_rr_prev, h_clock], dim=-1))

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        B, T, D = x.shape
        n_ep = T // self.episode_len
        ep_len = n_ep * self.episode_len
        ep_rhythm = x[:, :ep_len].reshape(B, n_ep, self.episode_len, D).mean(dim=2)

        return {
            "beat_rhythm": x,
            "episode_rhythm": ep_rhythm,
        }
