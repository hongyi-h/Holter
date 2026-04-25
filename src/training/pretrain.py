"""Pretraining loop for HolterFM with DDP multi-GPU support."""

from __future__ import annotations

import math
import os
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast

from src.models.holter_fm import HolterFM
from src.data.holter_dataset import HolterPretrainDataset, collate_holter
from src.data.holter_record import HolterRecord
from src.training.pretrain_losses import HolterFMPretrainLoss


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def log(msg: str):
    if is_main():
        print(msg, flush=True)


def build_optimizer(model: nn.Module, lr: float = 2e-4, weight_decay: float = 0.05):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "bias" in name or "norm" in name or "embedding" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=lr, betas=(0.9, 0.95))


def cosine_schedule(optimizer, step: int, total_steps: int, warmup: int = 2000,
                    lr_max: float = 2e-4, lr_min: float = 2e-5):
    if step < warmup:
        lr = lr_max * step / warmup
    else:
        progress = (step - warmup) / max(total_steps - warmup, 1)
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def pretrain(
    data_dir: str,
    output_dir: str = "checkpoints",
    epochs: int = 40,
    batch_size: int = 1,
    lr: float = 2e-4,
    grad_clip: float = 1.0,
    warmup_steps: int = 2000,
    save_every: int = 2,
    log_every: int = 10,
    use_amp: bool = True,
    num_workers: int = 4,
    resume_from: str | None = None,
):
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    use_ddp = world_size > 1

    output_dir = Path(output_dir)
    if is_main():
        output_dir.mkdir(parents=True, exist_ok=True)

    # --- data ---
    records = HolterRecord.discover(data_dir)
    log(f"Found {len(records)} recordings in {data_dir}")
    if len(records) == 0:
        raise ValueError(f"No recordings found in {data_dir}")

    # deterministic split
    rng = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(records), generator=rng).tolist()
    n = len(records)
    n_train = int(n * 0.7)
    n_val = int(n * 0.1)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]

    train_ds = HolterPretrainDataset(train_records, augment=True)
    val_ds = HolterPretrainDataset(val_records, augment=False)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if use_ddp else None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_holter, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_holter, num_workers=num_workers,
        pin_memory=True,
    )

    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    log(f"Train: {len(train_ds)}, Val: {len(val_ds)}, World: {world_size}")
    log(f"Steps/epoch: {steps_per_epoch}, Total steps: {total_steps}")
    log(f"Global batch size: {batch_size * world_size}")

    # --- model ---
    model = HolterFM().to(device)
    loss_fn = HolterFMPretrainLoss().to(device)

    if is_main():
        params = model.count_parameters()
        log(f"Model parameters: {json.dumps({k: f'{v/1e6:.1f}M' for k, v in params.items()})}")

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    all_params = list(model.parameters()) + list(loss_fn.parameters())
    optimizer = build_optimizer(model, lr=lr)
    # also add loss_fn params that need optimization
    loss_params = [p for p in loss_fn.parameters() if p.requires_grad]
    if loss_params:
        optimizer.add_param_group({"params": loss_params, "weight_decay": 0.0})

    scaler = GradScaler(enabled=use_amp)

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if resume_from:
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        raw_model = model.module if use_ddp else model
        raw_model.load_state_dict(ckpt["model"])
        loss_fn.load_state_dict(ckpt["loss_fn"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        log(f"Resumed from epoch {start_epoch}, step {global_step}")

    # --- training ---
    for epoch in range(start_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        loss_fn.train()
        epoch_losses = {}
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            lr_now = cosine_schedule(optimizer, global_step, total_steps, warmup_steps, lr)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = model(batch)
                losses = loss_fn(out, batch, epoch, epochs)

            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1

            for k, v in losses.items():
                epoch_losses.setdefault(k, 0.0)
                epoch_losses[k] += v.item()

            if step % log_every == 0:
                loss_str = " ".join(f"{k}={v.item():.4f}" for k, v in sorted(losses.items()) if k != "total")
                log(f"  [{epoch}/{epochs}] step {step}/{steps_per_epoch} lr={lr_now:.2e} "
                    f"total={losses['total'].item():.4f} gnorm={grad_norm:.2f} {loss_str}")

        # epoch summary
        dt = time.time() - t0
        avg = {k: v / max(steps_per_epoch, 1) for k, v in epoch_losses.items()}
        log(f"Epoch {epoch} ({dt:.0f}s): " + " ".join(f"{k}={v:.4f}" for k, v in sorted(avg.items())))

        # --- validation ---
        model.eval()
        loss_fn.eval()
        val_losses = {}
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    out = model(batch)
                    losses = loss_fn(out, batch, epoch, epochs)
                for k, v in losses.items():
                    val_losses.setdefault(k, 0.0)
                    val_losses[k] += v.item()

        n_val_steps = max(len(val_loader), 1)
        val_avg = {k: v / n_val_steps for k, v in val_losses.items()}

        # all-reduce val loss for consistent model selection
        if use_ddp:
            total_val = torch.tensor(val_avg.get("total", float("inf")), device=device)
            dist.all_reduce(total_val, op=dist.ReduceOp.AVG)
            val_avg["total"] = total_val.item()

        log(f"  Val: " + " ".join(f"{k}={v:.4f}" for k, v in sorted(val_avg.items())))

        # --- save (rank 0 only) ---
        if is_main():
            raw_model = model.module if use_ddp else model
            if epoch % save_every == 0 or epoch == epochs - 1:
                ckpt_path = output_dir / f"holter_fm_epoch{epoch:03d}.pt"
                torch.save({
                    "epoch": epoch,
                    "global_step": global_step,
                    "model": raw_model.state_dict(),
                    "loss_fn": loss_fn.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "train_losses": avg,
                    "val_losses": val_avg,
                    "best_val_loss": best_val_loss,
                }, ckpt_path)
                log(f"  Saved {ckpt_path}")

            if val_avg.get("total", float("inf")) < best_val_loss:
                best_val_loss = val_avg["total"]
                best_path = output_dir / "holter_fm_best.pt"
                torch.save({
                    "epoch": epoch,
                    "global_step": global_step,
                    "model": raw_model.state_dict(),
                    "loss_fn": loss_fn.state_dict(),
                    "best_val_loss": best_val_loss,
                }, best_path)
                log(f"  New best val loss: {best_val_loss:.4f}")

        if use_ddp:
            dist.barrier()

    if use_ddp:
        dist.destroy_process_group()
    log("Pretraining complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    pretrain(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume_from=args.resume,
        num_workers=args.num_workers,
    )
