"""Temporal order prediction loss with horizon weighting."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class OrderPredictionLoss(nn.Module):
    """Predict temporal distance between window pairs at multiple horizons."""

    def __init__(self, horizons=(1, 3, 6, 24, 72), horizon_weights=None):
        super().__init__()
        self.horizons = horizons
        if horizon_weights is None:
            horizon_weights = [1.0 / len(horizons)] * len(horizons)
        self.horizon_weights = horizon_weights

    def forward(self, predictions, targets, window_mask, quality_mask):
        """
        predictions: (B, W, embed_dim) - predicted window embeddings
        targets: (B, W, embed_dim) - target window embeddings (from EMA)
        window_mask: (B, W)
        quality_mask: (B, W)
        Returns: dict with l_order and per-horizon losses
        """
        combined_mask = window_mask & quality_mask  # (B, W)
        B, W, D = predictions.shape

        total_loss = torch.tensor(0.0, device=predictions.device)
        horizon_losses = {}

        for hi, h in enumerate(self.horizons):
            if h >= W:
                continue
            pred_i = predictions[:, :-h]  # (B, W-h, D)
            targ_j = targets[:, h:]       # (B, W-h, D)
            mask_i = combined_mask[:, :-h]
            mask_j = combined_mask[:, h:]
            pair_mask = mask_i & mask_j  # (B, W-h)

            if pair_mask.sum() == 0:
                continue

            # Cosine similarity as proxy for order
            cos_sim = F.cosine_similarity(pred_i, targ_j, dim=-1)  # (B, W-h)
            # Target: similarity should decrease with distance
            target_sim = 1.0 / (1.0 + h * 0.1)
            loss_h = F.huber_loss(cos_sim[pair_mask], torch.full_like(cos_sim[pair_mask], target_sim))

            horizon_losses[f"l_order_h{h}"] = loss_h
            total_loss = total_loss + self.horizon_weights[hi] * loss_h

        result = {"l_order": total_loss}
        result.update(horizon_losses)
        return result
