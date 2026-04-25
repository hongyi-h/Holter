"""Multi-scale self-supervised pretraining objectives for HolterFM."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BeatMAELoss(nn.Module):
    """Masked beat reconstruction: patchify → mask 50% → reconstruct."""

    def __init__(self, embed_dim: int = 256, patch_len: int = 4, n_channels: int = 3,
                 mask_ratio: float = 0.5, decoder_layers: int = 2):
        super().__init__()
        self.patch_len = patch_len
        self.n_channels = n_channels
        self.mask_ratio = mask_ratio
        n_patches = 80 // patch_len  # 20
        patch_dim = patch_len * n_channels

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, n_patches * patch_dim),
        )
        self.n_patches = n_patches
        self.patch_dim = patch_dim

    def forward(self, beat_embeds: torch.Tensor, windows: torch.Tensor,
                valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            beat_embeds: (B, T, 256)
            windows: (B, T, 80, 3)
            valid_mask: (B, T) bool
        """
        B, T, W, C = windows.shape
        # patchify: (B, T, 20, 12)
        patches = windows.reshape(B, T, self.n_patches, self.patch_dim)

        # reconstruct all patches from embedding
        recon = self.decoder(beat_embeds).reshape(B, T, self.n_patches, self.patch_dim)

        # generate mask: (B, T, 20)
        mask = torch.rand(B, T, self.n_patches, device=windows.device) < self.mask_ratio

        # L1 + derivative-L1 on masked patches only
        diff = (recon - patches).abs()
        masked = diff * mask.unsqueeze(-1).float()
        valid_expanded = valid_mask.unsqueeze(-1).unsqueeze(-1).float()
        l1 = (masked * valid_expanded).sum() / (mask.float() * valid_expanded.squeeze(-1)).sum().clamp(min=1)

        # derivative loss (temporal smoothness)
        recon_wave = recon.reshape(B, T, -1, C)
        target_wave = patches.reshape(B, T, -1, C)
        d_recon = recon_wave[:, :, 1:] - recon_wave[:, :, :-1]
        d_target = target_wave[:, :, 1:] - target_wave[:, :, :-1]
        d_loss = ((d_recon - d_target).abs() * valid_mask.unsqueeze(-1).unsqueeze(-1).float()).mean()

        return 0.8 * l1 + 0.2 * d_loss


class BeatNCELoss(nn.Module):
    """Contrastive loss between two augmented views of the same beat."""

    def __init__(self, embed_dim: int = 256, temperature: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 128),
        )
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """z1, z2: (N, embed_dim) — two views of the same beats."""
        h1 = F.normalize(self.proj(z1), dim=-1)
        h2 = F.normalize(self.proj(z2), dim=-1)
        logits = h1 @ h2.T / self.temperature
        labels = torch.arange(len(h1), device=h1.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


class EpisodeCPCLoss(nn.Module):
    """Contrastive predictive coding: predict next episode from current."""

    def __init__(self, d_model: int = 384, n_ahead: int = 2):
        super().__init__()
        self.n_ahead = n_ahead
        self.predictors = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
            for _ in range(n_ahead)
        ])
        self.temperature = 0.1

    def forward(self, episode_tokens: list[torch.Tensor], n_episodes: torch.Tensor) -> torch.Tensor:
        """episode_tokens: list of (n_ep, 384) per sample."""
        total_loss = 0.0
        count = 0
        for tokens in episode_tokens:
            n_ep = tokens.shape[0]
            if n_ep < 3:
                continue
            for k in range(self.n_ahead):
                if n_ep <= k + 1:
                    break
                pred = self.predictors[k](tokens[:n_ep - k - 1])
                target = tokens[k + 1:n_ep]
                pred = F.normalize(pred, dim=-1)
                target = F.normalize(target, dim=-1)
                logits = pred @ target.T / self.temperature
                labels = torch.arange(len(pred), device=pred.device)
                total_loss += F.cross_entropy(logits, labels)
                count += 1
        return total_loss / max(count, 1)


class EpisodeAlignLoss(nn.Module):
    """Align waveform episode token with rhythm episode token (symmetric InfoNCE)."""

    def __init__(self, wave_dim: int = 384, rhythm_dim: int = 128, proj_dim: int = 128,
                 temperature: float = 0.07):
        super().__init__()
        self.wave_proj = nn.Sequential(nn.Linear(wave_dim, proj_dim), nn.GELU(), nn.Linear(proj_dim, proj_dim))
        self.rhythm_proj = nn.Sequential(nn.Linear(rhythm_dim, proj_dim), nn.GELU(), nn.Linear(proj_dim, proj_dim))
        self.temperature = temperature

    def forward(self, ep_wave: torch.Tensor, ep_rhythm: torch.Tensor,
                n_episodes: torch.Tensor) -> torch.Tensor:
        """ep_wave: (B, max_ep, 384), ep_rhythm: (B, max_ep, 128)."""
        all_w, all_r = [], []
        for b in range(ep_wave.shape[0]):
            n = n_episodes[b].item()
            if n > 0:
                all_w.append(ep_wave[b, :n])
                all_r.append(ep_rhythm[b, :n])
        if not all_w:
            return torch.tensor(0.0, device=ep_wave.device)
        w = F.normalize(self.wave_proj(torch.cat(all_w)), dim=-1)
        r = F.normalize(self.rhythm_proj(torch.cat(all_r)), dim=-1)
        logits = w @ r.T / self.temperature
        labels = torch.arange(len(w), device=w.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


class EpisodeOrderLoss(nn.Module):
    """Binary: are two adjacent episodes in correct temporal order?"""

    def __init__(self, d_model: int = 384):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, episode_tokens: list[torch.Tensor]) -> torch.Tensor:
        pairs, labels = [], []
        for tokens in episode_tokens:
            n = tokens.shape[0]
            if n < 2:
                continue
            for _ in range(min(n - 1, 16)):
                i = torch.randint(0, n - 1, (1,)).item()
                if torch.rand(1).item() < 0.5:
                    pairs.append(torch.cat([tokens[i], tokens[i + 1]]))
                    labels.append(1.0)
                else:
                    pairs.append(torch.cat([tokens[i + 1], tokens[i]]))
                    labels.append(0.0)
        if not pairs:
            return torch.tensor(0.0, device=episode_tokens[0].device if episode_tokens else "cpu")
        pairs = torch.stack(pairs)
        labels = torch.tensor(labels, device=pairs.device)
        logits = self.classifier(pairs).squeeze(-1)
        return F.binary_cross_entropy_with_logits(logits, labels)


class DayMaskLoss(nn.Module):
    """Mask 15% of episode tokens before day encoder, reconstruct from day states."""

    def __init__(self, d_model: int = 512):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, episode_ctx: torch.Tensor, episode_targets: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        episode_ctx: (B, n_ep, 512) — day encoder output
        episode_targets: (B, n_ep, 512) — original fused episode tokens (before masking)
        mask: (B, n_ep) bool — True for masked positions
        """
        if mask.sum() == 0:
            return torch.tensor(0.0, device=episode_ctx.device)
        pred = self.decoder(episode_ctx)
        loss = F.mse_loss(pred[mask], episode_targets[mask])
        return loss


class DayStatsLoss(nn.Module):
    """Predict day-level statistics from day embedding."""

    def __init__(self, d_model: int = 512, n_stats: int = 12):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_stats),
        )
        self.register_buffer("running_mean", torch.zeros(n_stats))
        self.register_buffer("running_std", torch.ones(n_stats))
        self.initialized = False

    def update_stats(self, stats: torch.Tensor):
        if not self.initialized:
            self.running_mean.copy_(stats.mean(0))
            std = stats.std(0) if stats.shape[0] > 1 else torch.ones_like(stats[0])
            self.running_std.copy_(std.clamp(min=1e-3))
            self.initialized = True
        else:
            self.running_mean.lerp_(stats.mean(0), 0.01)
            if stats.shape[0] > 1:
                self.running_std.lerp_(stats.std(0).clamp(min=1e-3), 0.01)

    def forward(self, day_embed: torch.Tensor, day_stats: torch.Tensor) -> torch.Tensor:
        self.update_stats(day_stats.detach())
        targets = (day_stats - self.running_mean) / self.running_std
        pred = self.head(day_embed)
        return F.huber_loss(pred, targets)


class DayReportLoss(nn.Module):
    """Multi-label BCE for report concept prediction."""

    def __init__(self, d_model: int = 512, n_concepts: int = 19):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_concepts),
        )

    def forward(self, day_embed: torch.Tensor, concept_labels: torch.Tensor) -> torch.Tensor:
        logits = self.head(day_embed)
        return F.binary_cross_entropy_with_logits(logits, concept_labels)


class RhythmMaskLoss(nn.Module):
    """Mask 30% of rhythm tokens, predict VQ code + RR bins."""

    def __init__(self, d_model: int = 128, n_codes: int = 512, n_rr_bins: int = 32,
                 mask_ratio: float = 0.3, span_range: tuple = (5, 20)):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.span_range = span_range
        self.code_head = nn.Linear(d_model, n_codes)
        self.rr_head = nn.Linear(d_model, n_rr_bins)

    def forward(self, beat_rhythm: torch.Tensor, code_idx: torch.Tensor,
                rr_bins: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        B, T, D = beat_rhythm.shape
        # generate span masks
        mask = self._generate_span_mask(B, T, beat_rhythm.device, ~padding_mask)

        if mask.sum() == 0:
            return torch.tensor(0.0, device=beat_rhythm.device)

        code_logits = self.code_head(beat_rhythm)
        rr_logits = self.rr_head(beat_rhythm)

        code_loss = F.cross_entropy(code_logits[mask], code_idx[mask])
        rr_loss = F.cross_entropy(rr_logits[mask], rr_bins[mask])
        return code_loss + 0.5 * rr_loss

    def _generate_span_mask(self, B: int, T: int, device: torch.device,
                            valid: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        n_mask = int(T * self.mask_ratio)
        for b in range(B):
            masked = 0
            valid_len = valid[b].sum().item()
            while masked < n_mask and masked < valid_len:
                span = torch.randint(self.span_range[0], self.span_range[1] + 1, (1,)).item()
                start = torch.randint(0, max(valid_len - span, 1), (1,)).item()
                end = min(start + span, valid_len)
                mask[b, start:end] = True
                masked += end - start
        return mask & valid


class RRNextLoss(nn.Module):
    """Predict log(RR_next) from rhythm state."""

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.head = nn.Linear(d_model, 1)

    def forward(self, beat_rhythm: torch.Tensor, metadata: torch.Tensor,
                padding_mask: torch.Tensor) -> torch.Tensor:
        # metadata[:, :, 1] = log(rr_next)
        pred = self.head(beat_rhythm).squeeze(-1)
        target = metadata[:, :, 1]
        valid = ~padding_mask
        if valid.sum() == 0:
            return torch.tensor(0.0, device=beat_rhythm.device)
        return F.huber_loss(pred[valid], target[valid])


class HolterFMPretrainLoss(nn.Module):
    """Combines all pretraining objectives with fixed weights."""

    def __init__(self, n_concepts: int = 19):
        super().__init__()
        # beat-level
        self.beat_mae = BeatMAELoss()
        # episode-level
        self.ep_cpc = EpisodeCPCLoss()
        self.ep_align = EpisodeAlignLoss()
        self.ep_order = EpisodeOrderLoss()
        # day-level
        self.day_mask = DayMaskLoss()
        self.day_stats = DayStatsLoss()
        self.day_report = DayReportLoss(n_concepts=n_concepts)
        # rhythm-level
        self.rhythm_mask = RhythmMaskLoss()
        self.rr_next = RRNextLoss()

        # fixed weights per experiment plan
        self.w = {
            "beat": 0.35, "episode": 0.20, "day": 0.20, "rhythm": 0.20, "report": 0.05,
        }

    def forward(self, model_out: dict, batch: dict, epoch: int, max_epochs: int) -> dict[str, torch.Tensor]:
        losses = {}

        # beat-level
        losses["beat_mae"] = self.beat_mae(
            model_out["beat_embeds"], batch["windows"], batch["valid_mask"]
        )
        losses["beat"] = losses["beat_mae"]

        # episode-level
        losses["ep_cpc"] = self.ep_cpc(model_out["episode_tokens"], batch["n_episodes"])
        n_ep = batch["n_episodes"]
        max_ep = max(n_ep).item()
        if max_ep > 0 and model_out["episode_ctx"] is not None:
            ep_wave = torch.zeros(len(n_ep), max_ep, 384, device=batch["windows"].device)
            ep_rhythm = torch.zeros(len(n_ep), max_ep, 128, device=batch["windows"].device)
            for b in range(len(n_ep)):
                ne = n_ep[b].item()
                if ne > 0:
                    ep_wave[b, :ne] = model_out["episode_tokens"][b]
                    ep_rhythm[b, :ne] = model_out["beat_rhythm"][b, :ne * 64].reshape(ne, 64, -1).mean(1)
            losses["ep_align"] = self.ep_align(ep_wave, ep_rhythm, n_ep)
        else:
            losses["ep_align"] = torch.tensor(0.0, device=batch["windows"].device)
        losses["ep_order"] = self.ep_order(model_out["episode_tokens"])
        losses["episode"] = losses["ep_cpc"] + 0.5 * losses["ep_align"] + 0.1 * losses["ep_order"]

        # day-level (ramp from 0 to full weight in first 5 epochs)
        ramp = min(epoch / 5.0, 1.0)
        losses["day_stats"] = self.day_stats(model_out["day_embed"], batch["day_stats"])
        losses["day"] = losses["day_stats"]

        # report (if available)
        if "concept_labels" in batch:
            losses["report"] = self.day_report(model_out["day_embed"], batch["concept_labels"])
        else:
            losses["report"] = torch.tensor(0.0, device=batch["windows"].device)

        # rhythm-level
        losses["rhythm_mask"] = self.rhythm_mask(
            model_out["beat_rhythm"], model_out["code_idx"],
            batch["rr_bins"], batch["padding_mask"],
        )
        losses["rr_next"] = self.rr_next(
            model_out["beat_rhythm"], batch["metadata"], batch["padding_mask"],
        )
        losses["rhythm"] = losses["rhythm_mask"] + 0.2 * losses["rr_next"]

        # VQ loss
        losses["vq"] = model_out["vq_loss"]

        # total
        losses["total"] = (
            self.w["beat"] * losses["beat"]
            + self.w["episode"] * losses["episode"]
            + self.w["day"] * ramp * losses["day"]
            + self.w["rhythm"] * losses["rhythm"]
            + self.w["report"] * ramp * losses["report"]
            + 0.1 * losses["vq"]
        )

        return losses
