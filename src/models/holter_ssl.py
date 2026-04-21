"""HolterSSL: Hierarchical self-supervised model for 24h Holter ECG."""
import copy
import torch
import torch.nn as nn

from src.models.beat_encoder import BeatEncoder
from src.models.window_encoder import WindowEncoder
from src.models.day_encoder import DayEncoder, SnippetAggregator
from src.models.predictor import Predictor


class HolterSSL(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.mode = cfg.get("mode", "ordered")
        self.ema_tau = cfg.get("ema_tau", 0.996)
        self.mask_ratio = cfg.get("mask_ratio", 0.15)
        self.gradient_checkpoint = cfg.get("gradient_checkpoint", False)

        beat_dim = cfg.get("beat_dim", 128)
        window_dim = cfg.get("window_dim", 256)
        day_dim = cfg.get("day_dim", 256)

        self.beat_encoder = BeatEncoder(
            in_channels=cfg.get("in_channels", 3),
            embed_dim=beat_dim,
            beat_samples=cfg.get("beat_samples", 128),
        )
        self.window_encoder = WindowEncoder(
            beat_dim=beat_dim, embed_dim=window_dim,
            n_heads=cfg.get("w_heads", 4), n_layers=cfg.get("w_layers", 4),
            dropout=cfg.get("dropout", 0.1), max_beats=cfg.get("max_beats", 300),
        )

        # Target encoder (EMA) - permanently eval mode
        self.target_beat_encoder = copy.deepcopy(self.beat_encoder)
        self.target_window_encoder = copy.deepcopy(self.window_encoder)
        self.target_beat_encoder.eval()
        self.target_window_encoder.eval()
        for p in self.target_beat_encoder.parameters():
            p.requires_grad = False
        for p in self.target_window_encoder.parameters():
            p.requires_grad = False

        if self.mode == "snippet":
            self.day_encoder = SnippetAggregator(window_dim=window_dim)
        else:
            self.day_encoder = DayEncoder(
                window_dim=window_dim, hidden_dim=day_dim,
                n_layers=cfg.get("d_layers", 2), dropout=cfg.get("dropout", 0.1),
            )

        self.predictor = Predictor(
            input_dim=window_dim, hidden_dim=cfg.get("pred_hidden", 128),
            output_dim=window_dim,
        )

        # Day-level predictor for gradient flow through day_encoder
        self.day_predictor = nn.Sequential(
            nn.Linear(day_dim, day_dim), nn.LayerNorm(day_dim),
            nn.GELU(), nn.Linear(day_dim, window_dim),
        )

        self.recon_head = nn.Linear(
            beat_dim, cfg.get("beat_samples", 128) * cfg.get("in_channels", 3)
        )
        self.recon_channels = cfg.get("in_channels", 3)
        self.recon_samples = cfg.get("beat_samples", 128)

    def train(self, mode=True):
        super().train(mode)
        # Keep target encoders always in eval mode
        self.target_beat_encoder.eval()
        self.target_window_encoder.eval()
        return self

    @torch.no_grad()
    def update_ema(self):
        tau = self.ema_tau
        for p_o, p_t in zip(self.beat_encoder.parameters(), self.target_beat_encoder.parameters()):
            p_t.data.mul_(tau).add_(p_o.data, alpha=1 - tau)
        for p_o, p_t in zip(self.window_encoder.parameters(), self.target_window_encoder.parameters()):
            p_t.data.mul_(tau).add_(p_o.data, alpha=1 - tau)
        # EMA-update BN buffers too
        for (_, b_o), (_, b_t) in zip(self.beat_encoder.named_buffers(), self.target_beat_encoder.named_buffers()):
            b_t.data.copy_(tau * b_t.data + (1 - tau) * b_o.data)
        for (_, b_o), (_, b_t) in zip(self.window_encoder.named_buffers(), self.target_window_encoder.named_buffers()):
            b_t.data.copy_(tau * b_t.data + (1 - tau) * b_o.data)

    def encode_windows(self, beat_tensors, beat_masks, time_encodings, window_mask, use_target=False, return_beat_embeds=False):
        B, W, N, S, C = beat_tensors.shape
        be = self.target_beat_encoder if use_target else self.beat_encoder
        we = self.target_window_encoder if use_target else self.window_encoder

        # Batch all beats at once: (B*W*N, S, C)
        flat_beats = beat_tensors.reshape(B * W * N, S, C)

        if self.training and not use_target and self.gradient_checkpoint:
            flat_beat_emb = torch.utils.checkpoint.checkpoint(
                be, flat_beats, use_reentrant=False
            )
        else:
            flat_beat_emb = be(flat_beats)  # (B*W*N, D_beat)

        # Window encoding: (B*W, N, D_beat) -> (B*W, D_w)
        beat_emb = flat_beat_emb.reshape(B * W, N, -1)
        flat_masks = beat_masks.reshape(B * W, N)
        flat_time = time_encodings.reshape(B * W, -1)

        if self.training and not use_target and self.gradient_checkpoint:
            w_emb = torch.utils.checkpoint.checkpoint(
                we, beat_emb, flat_masks, flat_time, use_reentrant=False
            )
        else:
            w_emb = we(beat_emb, flat_masks, flat_time)

        window_out = w_emb.reshape(B, W, -1)
        if return_beat_embeds:
            return window_out, flat_beat_emb.reshape(B, W, N, -1)
        return window_out

    def _random_mask_beats(self, beat_tensors):
        B, W, N, S, C = beat_tensors.shape
        mask = torch.rand(B, W, N, device=beat_tensors.device) < self.mask_ratio
        target_beats = beat_tensors.clone()
        masked_beats = beat_tensors.clone()
        masked_beats[mask] = 0.0
        return masked_beats, target_beats, mask

    def forward(self, batch):
        beat_tensors = batch["beat_tensors"]
        beat_masks = batch["beat_masks"]
        time_encodings = batch["time_encodings"]
        quality_mask = batch["quality_mask"]
        window_mask = batch["window_mask"]

        if self.training:
            masked_beats, target_beats, beat_mask_recon = self._random_mask_beats(beat_tensors)
            # return_beat_embeds=True so we can reuse for reconstruction (avoids 2nd OOM-prone pass)
            window_embeds, beat_embeds = self.encode_windows(masked_beats, beat_masks, time_encodings, window_mask, use_target=False, return_beat_embeds=True)
            with torch.no_grad():
                target_window_embeds = self.encode_windows(beat_tensors, beat_masks, time_encodings, window_mask, use_target=True)
            predictions = self.predictor(window_embeds)
            day_embed, window_outputs = self.day_encoder(window_embeds, window_mask)
            day_prediction = self.day_predictor(day_embed)
            B, W, N, S, C = beat_tensors.shape
            recon_flat = self.recon_head(beat_embeds.reshape(B * W * N, -1))
            recon = recon_flat.reshape(B, W, N, S, C)
            return {
                "window_embeds": window_embeds,
                "target_window_embeds": target_window_embeds,
                "predictions": predictions,
                "day_embed": day_embed,
                "day_prediction": day_prediction,
                "window_outputs": window_outputs,
                "window_mask": window_mask,
                "quality_mask": quality_mask,
                "recon": recon,
                "target_beats": target_beats,
                "beat_mask_recon": beat_mask_recon,
                "beat_masks": beat_masks,
            }
        else:
            # Inference: no masking
            window_embeds = self.encode_windows(beat_tensors, beat_masks, time_encodings, window_mask, use_target=False)
            day_embed, window_outputs = self.day_encoder(window_embeds, window_mask)
            return {
                "window_embeds": window_embeds,
                "day_embed": day_embed,
                "window_outputs": window_outputs,
                "window_mask": window_mask,
                "quality_mask": quality_mask,
            }
