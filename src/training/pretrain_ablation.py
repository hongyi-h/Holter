"""Pretraining with ablation variants for M4 experiments."""

from __future__ import annotations

import argparse
from src.training.pretrain import pretrain, setup_distributed, is_main, log
from src.models.holter_fm import HolterFM
from src.models.episode_encoder import EpisodeEncoder
from src.models.day_encoder import DayEncoder
from src.models.rhythm_branch import RhythmBranch
from src.training.pretrain_losses import HolterFMPretrainLoss


ABLATION_CONFIGS = {
    "no_beat_sync": {
        "description": "A1: Fixed 10s chunk tokenization instead of beat-synchronous",
    },
    "no_episode_loss": {
        "description": "A2: Remove episode-level losses (CPC, align, order)",
        "loss_zero": ["ep_cpc", "ep_align", "ep_order"],
    },
    "no_day_loss": {
        "description": "A3: Remove day-level losses (mask, stats)",
        "loss_zero": ["day_stats"],
        "loss_weight_override": {"day": 0.0},
    },
    "no_rhythm": {
        "description": "A4: Remove rhythm branch entirely",
        "remove_rhythm": True,
    },
    "beat_only_ssl": {
        "description": "A5: Beat-only SSL, no episode/day losses",
        "loss_weight_override": {"episode": 0.0, "day": 0.0, "report": 0.0},
    },
    "no_day_aux": {
        "description": "A6: Remove report + stats auxiliary losses",
        "loss_weight_override": {"report": 0.0},
        "loss_zero": ["day_stats"],
    },
    "transformer_day": {
        "description": "A7: Sparse Transformer day encoder instead of Mamba",
        "use_transformer_day": True,
    },
    "segment_model": {
        "description": "A8: Parameter-matched segment model (no full-day context)",
        "no_day_encoder": True,
    },
}


def apply_ablation_to_loss(loss_fn: HolterFMPretrainLoss, config: dict):
    """Modify loss weights/components based on ablation config."""
    if "loss_weight_override" in config:
        for k, v in config["loss_weight_override"].items():
            loss_fn.w[k] = v

    return loss_fn


def main():
    parser = argparse.ArgumentParser(description="Ablation pretraining")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints/ablations")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ablation", type=str, required=True, choices=list(ABLATION_CONFIGS.keys()))
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = ABLATION_CONFIGS[args.ablation]
    if is_main():
        print(f"Ablation: {args.ablation}")
        print(f"  {config['description']}")

    pretrain(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume_from=args.resume,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
