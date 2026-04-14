"""Anti-collapse regularization: variance floor + covariance decorrelation."""
import torch
import torch.nn as nn


class AntiCollapseLoss(nn.Module):
    """VICReg-style regularization to prevent embedding collapse."""

    def __init__(self, var_weight=1.0, cov_weight=0.04, var_floor=1.0):
        super().__init__()
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.var_floor = var_floor

    def forward(self, embeddings, mask=None):
        """
        embeddings: (B, D) or (B, W, D)
        mask: optional (B,) or (B, W) bool
        Returns: dict with l_var, l_cov
        """
        if embeddings.dim() == 3:
            B, W, D = embeddings.shape
            if mask is not None:
                flat_mask = mask.reshape(-1)
                emb = embeddings.reshape(-1, D)[flat_mask]
            else:
                emb = embeddings.reshape(-1, D)
        else:
            emb = embeddings
            if mask is not None:
                emb = emb[mask]

        if emb.shape[0] < 2:
            return {"l_var": torch.tensor(0.0, device=emb.device),
                    "l_cov": torch.tensor(0.0, device=emb.device)}

        # Variance loss: encourage each dimension to have variance >= var_floor
        std = emb.std(dim=0)
        l_var = torch.relu(self.var_floor - std).mean()

        # Covariance loss: decorrelate dimensions
        emb_centered = emb - emb.mean(dim=0, keepdim=True)
        cov = (emb_centered.T @ emb_centered) / (emb.shape[0] - 1)
        # Zero out diagonal
        D = cov.shape[0]
        off_diag = cov.flatten()[:-1].view(D - 1, D + 1)[:, 1:].flatten()
        l_cov = (off_diag ** 2).mean()

        return {
            "l_var": self.var_weight * l_var,
            "l_cov": self.cov_weight * l_cov,
        }
