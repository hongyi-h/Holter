"""HolterFM: full model combining beat tokenizer, episode encoder, rhythm branch, day encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .beat_tokenizer import BeatTokenizer
from .episode_encoder import EpisodeEncoder
from .rhythm_branch import RhythmBranch
from .day_encoder import DayEncoder


class HolterFM(nn.Module):
    """42.9M-parameter Holter Foundation Model.

    Forward pass processes one full day:
    1. Beat tokenizer: (n_beats, 80, 3) → (n_beats, 256) embeddings + VQ codes
    2. Episode encoder: groups of 64 beats → (n_episodes, 384) episode tokens
    3. Rhythm branch: (n_beats,) rhythm tokens → (n_beats, 128) + (n_episodes, 128)
    4. Day encoder: fused episodes → (512,) day embedding
    """

    def __init__(
        self,
        beat_embed_dim: int = 256,
        episode_d_model: int = 384,
        rhythm_d_model: int = 128,
        day_d_model: int = 512,
        episode_len: int = 64,
        n_codes: int = 512,
        n_rr_bins: int = 32,
    ):
        super().__init__()
        self.episode_len = episode_len
        self.day_d_model = day_d_model

        self.beat_tokenizer = BeatTokenizer(
            embed_dim=beat_embed_dim, code_dim=64, n_codes=n_codes,
        )
        self.episode_encoder = EpisodeEncoder(
            beat_dim=beat_embed_dim, d_model=episode_d_model, episode_len=episode_len,
        )
        self.rhythm_branch = RhythmBranch(
            n_codes=n_codes, n_rr_bins=n_rr_bins,
            d_model=rhythm_d_model, episode_len=episode_len,
        )
        self.day_encoder = DayEncoder(
            episode_waveform_dim=episode_d_model,
            episode_rhythm_dim=rhythm_d_model,
            d_model=day_d_model,
        )

        # beat-level representation for downstream tasks
        self.beat_proj = nn.Sequential(
            nn.Linear(beat_embed_dim + episode_d_model + rhythm_d_model, day_d_model),
            nn.GELU(),
            nn.Linear(day_d_model, day_d_model),
        )

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        """Process a batch of full-day recordings.

        Expects batch from collate_holter with keys:
            windows: (B, max_beats, 80, 3)
            metadata: (B, max_beats, 6)
            n_beats: (B,)
            n_episodes: (B,)
            rr_bins, rr_prev_bins, hour_sin, hour_cos: (B, max_beats)
            padding_mask: (B, max_beats) — True for padded positions
        """
        B = batch["windows"].shape[0]
        device = batch["windows"].device
        max_beats = batch["windows"].shape[1]

        # --- 1. Beat tokenizer (process all beats flat) ---
        flat_wins = batch["windows"].reshape(-1, batch["windows"].shape[2], 3)
        flat_meta = batch["metadata"].reshape(-1, 6)
        beat_out = self.beat_tokenizer(flat_wins, flat_meta)
        beat_embeds = beat_out["embed"].reshape(B, max_beats, -1)
        code_idx = beat_out["code_idx"].reshape(B, max_beats)
        vq_loss = beat_out["vq_loss"]

        # --- 2. Episode encoder (process episodes per sample) ---
        ep_len = self.episode_len
        all_episode_tokens = []
        all_beat_ctx = []

        for b in range(B):
            n_b = batch["n_beats"][b].item()
            n_ep = batch["n_episodes"][b].item()
            ep_beats = n_ep * ep_len

            if n_ep == 0:
                all_episode_tokens.append(torch.zeros(0, self.episode_encoder.d_model, device=device))
                all_beat_ctx.append(torch.zeros(n_b, self.episode_encoder.d_model, device=device))
                continue

            ep_input = beat_embeds[b, :ep_beats].reshape(n_ep, ep_len, -1)
            ep_out = self.episode_encoder(ep_input)
            all_episode_tokens.append(ep_out["episode_token"])  # (n_ep, 384)

            ctx = ep_out["beat_ctx"].reshape(ep_beats, -1)  # (ep_beats, 384)
            if n_b > ep_beats:
                pad = torch.zeros(n_b - ep_beats, ctx.shape[-1], device=device)
                ctx = torch.cat([ctx, pad], dim=0)
            all_beat_ctx.append(ctx[:n_b])

        # --- 3. Rhythm branch ---
        rhythm_out = self.rhythm_branch(
            code_idx=code_idx,
            rr_bins=batch["rr_bins"],
            rr_prev_bins=batch["rr_prev_bins"],
            hour_sin=batch["hour_sin"],
            hour_cos=batch["hour_cos"],
        )

        # --- 4. Day encoder (with episode masking for pretraining) ---
        max_ep = max(batch["n_episodes"]).item()
        ep_fused_targets = None
        ep_mask = None
        if max_ep == 0:
            day_embed = torch.zeros(B, self.day_d_model, device=device)
            episode_ctx = None
        else:
            ep_wave = torch.zeros(B, max_ep, self.episode_encoder.d_model, device=device)
            ep_rhythm = torch.zeros(B, max_ep, self.rhythm_branch.d_model, device=device)
            for b in range(B):
                n_ep = batch["n_episodes"][b].item()
                if n_ep > 0:
                    ep_wave[b, :n_ep] = all_episode_tokens[b]
                    ep_rhythm[b, :n_ep] = rhythm_out["episode_rhythm"][b, :n_ep]

            day_out = self.day_encoder(ep_wave, ep_rhythm)
            day_embed = day_out["day_embed"]
            episode_ctx = day_out["episode_ctx"]
            ep_fused_targets = day_out.get("_fused_input")

        # --- 5. Beat-level representation ---
        beat_rhythm = rhythm_out["beat_rhythm"]  # (B, max_beats, 128)
        beat_ctx_padded = torch.zeros(B, max_beats, self.episode_encoder.d_model, device=device)
        for b in range(B):
            n_b = min(batch["n_beats"][b].item(), max_beats)
            beat_ctx_padded[b, :n_b] = all_beat_ctx[b][:n_b]

        beat_repr = self.beat_proj(
            torch.cat([beat_embeds, beat_ctx_padded, beat_rhythm], dim=-1)
        )

        return {
            "beat_embeds": beat_embeds,       # (B, max_beats, 256)
            "beat_repr": beat_repr,           # (B, max_beats, 512)
            "code_idx": code_idx,             # (B, max_beats)
            "episode_tokens": all_episode_tokens,  # list of (n_ep, 384)
            "episode_ctx": episode_ctx,       # (B, max_ep, 512) or None
            "ep_fused_targets": ep_fused_targets,  # (B, max_ep, 512) detached, for DayMaskLoss
            "beat_rhythm": beat_rhythm,       # (B, max_beats, 128)
            "day_embed": day_embed,           # (B, 512)
            "vq_loss": vq_loss,               # scalar
        }

    def count_parameters(self) -> dict[str, int]:
        counts = {}
        for name, module in [
            ("beat_tokenizer", self.beat_tokenizer),
            ("episode_encoder", self.episode_encoder),
            ("rhythm_branch", self.rhythm_branch),
            ("day_encoder", self.day_encoder),
            ("beat_proj", self.beat_proj),
        ]:
            counts[name] = sum(p.numel() for p in module.parameters())
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts
