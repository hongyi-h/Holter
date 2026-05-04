"""Downstream evaluation heads for HolterFM."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BeatClassificationHead(nn.Module):
    """Task 1: Beat classification (N/V/F) from contextualized beat representation."""

    def __init__(self, d_model: int = 512, n_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, beat_repr: torch.Tensor) -> torch.Tensor:
        return self.head(beat_repr)  # (B, T, 3)

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor,
                     valid_mask: torch.Tensor, class_weights: torch.Tensor | None = None) -> torch.Tensor:
        # logits: (B, T, 3), labels: (B, T), valid_mask: (B, T)
        B, T, C = logits.shape
        known_mask = valid_mask & (labels >= 0) & (labels < C)
        logits_flat = logits[known_mask]
        labels_flat = labels[known_mask]
        if labels_flat.numel() == 0:
            return torch.tensor(0.0, device=logits.device)
        if class_weights is None:
            class_weights = torch.tensor([1.0, 20.0, 50.0], device=logits.device)
        return F.cross_entropy(logits_flat, labels_flat, weight=class_weights)


class PVCBurdenHead(nn.Module):
    """Task 2: PVC burden regression from day embedding + hourly pooled embeddings."""

    def __init__(self, d_model: int = 512, n_hourly: int = 24):
        super().__init__()
        self.hourly_pool = nn.Linear(d_model, d_model // 4)
        self.regressor = nn.Sequential(
            nn.Linear(d_model + n_hourly * (d_model // 4), d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 2),  # [burden_pct, log1p_count]
        )
        self.n_hourly = n_hourly

    def forward(self, day_embed: torch.Tensor, episode_ctx: torch.Tensor,
                n_episodes: torch.Tensor) -> torch.Tensor:
        B = day_embed.shape[0]
        device = day_embed.device
        hourly = torch.zeros(B, self.n_hourly, episode_ctx.shape[-1] // 4, device=device)

        for b in range(B):
            n_ep = n_episodes[b].item()
            if n_ep == 0:
                continue
            ep_feats = self.hourly_pool(episode_ctx[b, :n_ep])  # (n_ep, d//4)
            # distribute episodes into 24 hourly bins
            bin_size = max(n_ep // self.n_hourly, 1)
            for h in range(self.n_hourly):
                start = h * bin_size
                end = min(start + bin_size, n_ep)
                if start < n_ep:
                    hourly[b, h] = ep_feats[start:end].mean(0)

        hourly_flat = hourly.reshape(B, -1)
        x = torch.cat([day_embed, hourly_flat], dim=-1)
        return self.regressor(x)  # (B, 2)

    def compute_loss(self, pred: torch.Tensor, day_stats: torch.Tensor) -> torch.Tensor:
        burden_target = day_stats[:, 7]  # pvc_burden
        count_target = torch.log1p(day_stats[:, 6])  # log1p(pvc_count)
        targets = torch.stack([burden_target, count_target], dim=-1)
        return F.huber_loss(pred, targets)


class ReportConceptHead(nn.Module):
    """Task 3: Multi-label report concept prediction from day embedding."""

    def __init__(self, d_model: int = 512, n_concepts: int = 19):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, n_concepts),
        )

    def forward(self, day_embed: torch.Tensor) -> torch.Tensor:
        return self.head(day_embed)  # (B, n_concepts)

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, labels)


class VentricularEventHead(nn.Module):
    """Task 5: Ventricular rhythm event tagging per episode.

    Labels: bigeminy, trigeminy, couplet, v_run (multi-label per episode).
    """

    def __init__(self, d_model: int = 512, n_events: int = 4):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, n_events),
        )

    def forward(self, episode_ctx: torch.Tensor) -> torch.Tensor:
        return self.head(episode_ctx)  # (B, n_ep, 4)

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
        # logits: (B, n_ep, 4), labels: (B, n_ep, 4), mask: (B, n_ep)
        logits_flat = logits[mask]
        labels_flat = labels[mask]
        if logits_flat.numel() == 0:
            return torch.tensor(0.0, device=logits.device)
        return F.binary_cross_entropy_with_logits(logits_flat, labels_flat)
