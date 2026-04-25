"""Beat tokenizer: Conv1d encoder + VQ codebook → 256-d beat embedding + discrete code."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, ch: int, r: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // r),
            nn.ReLU(inplace=True),
            nn.Linear(ch // r, ch),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        w = x.mean(dim=-1)
        w = self.fc(w).unsqueeze(-1)
        return x * w


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, use_se: bool = False):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        h = self.se(h)
        return F.gelu(h + self.skip(x))


class AttentivePool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) → pool over T
        w = self.attn(x.transpose(1, 2)).squeeze(-1)  # (B, T)
        w = F.softmax(w, dim=-1).unsqueeze(1)          # (B, 1, T)
        return (x * w).sum(dim=-1)                      # (B, C)


class VQCodebook(nn.Module):
    def __init__(self, n_codes: int = 512, code_dim: int = 64, commitment: float = 0.25):
        super().__init__()
        self.n_codes = n_codes
        self.code_dim = code_dim
        self.commitment = commitment
        self.embedding = nn.Embedding(n_codes, code_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / n_codes, 1.0 / n_codes)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z: (B, code_dim)
        # ||z - e||² = ||z||² + ||e||² - 2*z·eᵀ  (avoids O(N×K×D) intermediate)
        dist = (z.pow(2).sum(-1, keepdim=True)
                + self.embedding.weight.pow(2).sum(-1).unsqueeze(0)
                - 2 * z @ self.embedding.weight.t())
        indices = dist.argmin(dim=-1)  # (B,)
        z_q = self.embedding(indices)
        # straight-through
        z_q_st = z + (z_q - z).detach()
        vq_loss = F.mse_loss(z_q.detach(), z) * self.commitment + F.mse_loss(z_q, z.detach())
        return z_q_st, indices, vq_loss


class BeatTokenizer(nn.Module):
    """Encodes a single beat window (80 samples × 3 channels) into a 256-d embedding + VQ code.

    Architecture: Conv1d stem → 4 ResBlocks → attentive pool → metadata MLP → projection.
    ~2.2M parameters.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 256,
        code_dim: int = 64,
        n_codes: int = 512,
        meta_dim: int = 6,
        meta_hidden: int = 32,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # stem: (B, 3, 80) → (B, 32, 40)
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )

        # residual blocks
        self.blocks = nn.Sequential(
            ResBlock(32, 64, kernel=7, use_se=False),
            ResBlock(64, 128, kernel=5, use_se=True),
            ResBlock(128, 192, kernel=5, use_se=True),
            ResBlock(192, 256, kernel=3, use_se=True),
        )

        self.pool = AttentivePool(256)

        # metadata MLP
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, meta_hidden),
            nn.GELU(),
            nn.Linear(meta_hidden, meta_hidden),
        )

        # final projection
        self.proj = nn.Linear(256 + meta_hidden, embed_dim)

        # VQ codebook
        self.code_proj = nn.Linear(256, code_dim)
        self.vq = VQCodebook(n_codes=n_codes, code_dim=code_dim)

    def forward(
        self, waveform: torch.Tensor, metadata: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            waveform: (B, window_len, 3) — beat-centered ECG window
            metadata: (B, 6) — [log_rr_prev, log_rr_next, inst_hr, hour_sin, hour_cos, norm_idx]

        Returns:
            dict with keys: embed (B, 256), code_embed (B, 64), code_idx (B,), vq_loss (scalar)
        """
        x = waveform.transpose(1, 2)  # (B, 3, 80)
        x = self.stem(x)               # (B, 32, 40)
        x = self.blocks(x)             # (B, 256, 40)
        h_wave = self.pool(x)          # (B, 256)

        h_meta = self.meta_mlp(metadata)  # (B, 32)
        h = self.proj(torch.cat([h_wave, h_meta], dim=-1))  # (B, 256)

        z_code = self.code_proj(h_wave)
        code_embed, code_idx, vq_loss = self.vq(z_code)

        return {
            "embed": h,
            "wave_embed": h_wave,
            "code_embed": code_embed,
            "code_idx": code_idx,
            "vq_loss": vq_loss,
        }
