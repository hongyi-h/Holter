"""Verify model builds correctly and runs a forward pass on sample data."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np

from src.data.holter_record import HolterRecord
from src.data.holter_dataset import HolterPretrainDataset, collate_holter
from src.models.holter_fm import HolterFM
from src.training.pretrain_losses import HolterFMPretrainLoss


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # --- Check mamba-ssm backend ---
    from src.models.rhythm_branch import HAS_MAMBA_SSM, _OfficialMamba
    print(f"mamba-ssm loaded: {HAS_MAMBA_SSM}")
    if HAS_MAMBA_SSM:
        print(f"  Backend class: {_OfficialMamba}")
    else:
        print("  WARNING: Using fallback SSM — will OOM on full-day sequences!")
        print("  Fix: pip install mamba-ssm (or check import errors)")
        # Try to diagnose why import failed
        try:
            import mamba_ssm
            print(f"  mamba_ssm package found at: {mamba_ssm.__file__}")
            print(f"  mamba_ssm version: {getattr(mamba_ssm, '__version__', 'unknown')}")
            print(f"  Available modules: {dir(mamba_ssm)}")
        except ImportError as e:
            print(f"  mamba_ssm not installed: {e}")

    # --- Load sample data ---
    records = HolterRecord.discover("data/DMS")
    print(f"Records: {len(records)}")

    ds = HolterPretrainDataset(records, augment=False)
    print(f"Dataset size: {len(ds)}")

    # Use a small subset of beats for testing (full day is too large for MPS/CPU)
    batch = ds[0]
    max_test_beats = 512  # limit for local testing
    n_beats = min(batch["n_beats"], max_test_beats)
    n_episodes = n_beats // 64

    mini_batch = {
        "record_id": [batch["record_id"]],
        "n_beats": torch.tensor([n_beats]),
        "n_episodes": torch.tensor([n_episodes]),
        "windows": batch["windows"][:n_beats].unsqueeze(0),
        "metadata": batch["metadata"][:n_beats].unsqueeze(0),
        "beat_labels": batch["beat_labels"][:n_beats].unsqueeze(0),
        "valid_mask": batch["valid_mask"][:n_beats].unsqueeze(0),
        "rr_bins": batch["rr_bins"][:n_beats].unsqueeze(0),
        "rr_prev_bins": batch["rr_prev_bins"][:n_beats].unsqueeze(0),
        "hour_sin": batch["hour_sin"][:n_beats].unsqueeze(0),
        "hour_cos": batch["hour_cos"][:n_beats].unsqueeze(0),
        "day_stats": batch["day_stats"].unsqueeze(0),
        "padding_mask": torch.zeros(1, n_beats, dtype=torch.bool),
    }
    if "concept_labels" in batch:
        mini_batch["concept_labels"] = batch["concept_labels"].unsqueeze(0)

    print(f"\nTest batch: {n_beats} beats, {n_episodes} episodes")
    print(f"  windows: {mini_batch['windows'].shape}")
    print(f"  metadata: {mini_batch['metadata'].shape}")

    # --- Build model ---
    model = HolterFM()
    params = model.count_parameters()
    print(f"\nModel parameters:")
    for k, v in params.items():
        print(f"  {k}: {v:,} ({v/1e6:.1f}M)")

    # --- Forward pass ---
    model = model.to(device)
    mini_batch_dev = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in mini_batch.items()
    }

    model.eval()
    print("\nRunning forward pass...")
    t0 = time.time()
    with torch.no_grad():
        out = model(mini_batch_dev)
    dt = time.time() - t0
    print(f"Forward pass: {dt:.2f}s")

    print(f"\nOutputs:")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape} (dtype={v.dtype})")
        elif isinstance(v, list):
            print(f"  {k}: list of {len(v)} tensors, first shape={v[0].shape if v else 'empty'}")
        else:
            print(f"  {k}: {v}")

    # --- Loss computation ---
    loss_fn = HolterFMPretrainLoss()
    loss_fn = loss_fn.to(device)
    print("\nComputing losses...")
    t0 = time.time()
    with torch.no_grad():
        losses = loss_fn(out, mini_batch_dev, epoch=0, max_epochs=40)
    dt = time.time() - t0
    print(f"Loss computation: {dt:.2f}s")
    print(f"\nLosses:")
    for k, v in sorted(losses.items()):
        print(f"  {k}: {v.item():.4f}")

    print("\n=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()
