"""Masked reconstruction loss (time + spectral domain)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedReconLoss(nn.Module):
    """Combined time-domain (smooth L1) and spectral-domain reconstruction loss.
    Only computed over actually masked and valid beats."""

    def __init__(self, spectral_weight=0.1):
        super().__init__()
        self.spectral_weight = spectral_weight

    def forward(self, recon, target, beat_mask_recon=None, beat_masks=None, window_mask=None):
        """
        recon: (B, W, N, S, C) reconstructed beats
        target: (B, W, N, S, C) original beats
        beat_mask_recon: (B, W, N) bool - True where beats were masked
        beat_masks: (B, W, N) bool - True where beats are valid (not padding)
        window_mask: (B, W) bool - True for valid windows
        """
        if beat_mask_recon is not None and beat_masks is not None:
            valid = beat_mask_recon & beat_masks
            if window_mask is not None:
                valid = valid & window_mask.unsqueeze(-1)
            valid_full = valid.unsqueeze(-1).unsqueeze(-1).expand_as(recon)
            n_valid = valid_full.sum()
            if n_valid == 0:
                zero = torch.tensor(0.0, device=recon.device)
                return {"l_masked": zero, "l_recon_time": zero, "l_recon_spec": zero}
            l_time = F.smooth_l1_loss(recon[valid_full], target[valid_full])
            recon_masked = recon[valid].reshape(-1, recon.shape[-2], recon.shape[-1])
            target_masked = target[valid].reshape(-1, target.shape[-2], target.shape[-1])
            recon_spec = torch.fft.rfft(recon_masked, dim=-2).abs()
            target_spec = torch.fft.rfft(target_masked, dim=-2).abs()
            l_spec = F.l1_loss(recon_spec, target_spec)
        else:
            l_time = F.smooth_l1_loss(recon, target)
            recon_spec = torch.fft.rfft(recon, dim=-2).abs()
            target_spec = torch.fft.rfft(target, dim=-2).abs()
            l_spec = F.l1_loss(recon_spec, target_spec)

        l_total = l_time + self.spectral_weight * l_spec
        return {"l_masked": l_total, "l_recon_time": l_time, "l_recon_spec": l_spec}
