"""SSL pre-training loop for HolterSSL with DeepSpeed + wandb."""
import os, sys, argparse, yaml, json, time, math
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import deepspeed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.holter_ssl import HolterSSL
from src.losses.masked_recon import MaskedReconLoss
from src.losses.order_prediction import OrderPredictionLoss
from src.losses.regularization import AntiCollapseLoss
from src.data.dataset import HolterPretrainDataset, collate_pretrain

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--local_rank", type=int, default=-1, help="Set by deepspeed launcher")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    loss_cfg = cfg["loss"]
    out_cfg = cfg["output"]
    ds_cfg_path = cfg.get("deepspeed_config", "configs/ds_config.json")

    torch.manual_seed(train_cfg["seed"])
    np.random.seed(train_cfg["seed"])

    # ---- Dataset ----
    dataset = HolterPretrainDataset(
        cfg["data"]["processed_dir"],
        mode=model_cfg["mode"],
        max_windows=train_cfg.get("max_windows", 288),
        max_beats=model_cfg.get("max_beats", 300),
    )

    # ---- Model ----
    model = HolterSSL(model_cfg)

    # ---- Losses (always on device, moved after engine init) ----
    l_masked = MaskedReconLoss(spectral_weight=loss_cfg["masked"]["spectral_weight"])
    l_order = OrderPredictionLoss(
        horizons=loss_cfg["order"]["horizons"],
        horizon_weights=loss_cfg["order"]["horizon_weights"],
    )
    l_reg = AntiCollapseLoss(
        var_weight=loss_cfg["regularization"]["var_weight"],
        cov_weight=loss_cfg["regularization"]["cov_weight"],
        var_floor=loss_cfg["regularization"].get("var_floor", 1.0),
    )
    w_cfg = loss_cfg["weights"]

    # ---- Patch DeepSpeed config with training params ----
    with open(ds_cfg_path) as f:
        ds_config = json.load(f)

    total_steps = train_cfg["epochs"] * (len(dataset) // train_cfg["batch_size"] + 1)
    warmup_steps = train_cfg.get("warmup_steps", 500)

    ds_config["train_micro_batch_size_per_gpu"] = train_cfg["batch_size"]
    if "optimizer" in ds_config and ds_config["optimizer"].get("params", {}).get("lr") == "auto":
        ds_config["optimizer"]["params"]["lr"] = train_cfg["lr"]
    if "scheduler" in ds_config:
        sched_p = ds_config["scheduler"].get("params", {})
        if sched_p.get("warmup_max_lr") == "auto":
            sched_p["warmup_max_lr"] = train_cfg["lr"]
        if sched_p.get("warmup_num_steps") == "auto":
            sched_p["warmup_num_steps"] = warmup_steps
        if sched_p.get("total_num_steps") == "auto":
            sched_p["total_num_steps"] = total_steps

    # ---- DeepSpeed init ----
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        args=args,
        model=model,
        config=ds_config,
    )
    device = model_engine.local_rank

    # Move losses to device
    l_masked = l_masked.to(device) if hasattr(l_masked, 'to') else l_masked
    l_order = l_order.to(device) if hasattr(l_order, 'to') else l_order

    # ---- DataLoader with DistributedSampler ----
    if dist.is_initialized():
        sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(),
                                     rank=dist.get_rank(), shuffle=True,
                                     seed=train_cfg["seed"])
        loader = DataLoader(dataset, batch_size=train_cfg["batch_size"],
                            sampler=sampler, collate_fn=collate_pretrain, num_workers=2,
                            pin_memory=True)
    else:
        sampler = None
        loader = DataLoader(dataset, batch_size=train_cfg["batch_size"],
                            shuffle=True, collate_fn=collate_pretrain, num_workers=0)

    if is_main_process():
        print(f"Device: cuda:{device}")
        print(f"Dataset: {len(dataset)} patients, {len(loader)} batches")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model params: {total_params:,}")
        print(f"DeepSpeed ZeRO stage: {ds_config.get('zero_optimization', {}).get('stage', 0)}")
        print(f"BF16: {ds_config.get('bf16', {}).get('enabled', False)}")
        print(f"FP16: {ds_config.get('fp16', {}).get('enabled', False)}")
        print(f"Total steps: {total_steps}, Warmup: {warmup_steps}")

    # ---- wandb init (rank 0 only) ----
    if HAS_WANDB and is_main_process():
        wandb_cfg = ds_config.get("wandb", {})
        wandb.init(
            project=wandb_cfg.get("project", "holter-ssl"),
            group=wandb_cfg.get("group", "pretrain"),
            name=f"{model_cfg['mode']}_seed{train_cfg['seed']}",
            config={
                "model": model_cfg,
                "training": train_cfg,
                "loss": loss_cfg,
                "deepspeed": {k: v for k, v in ds_config.items()
                              if k not in ("wandb",)},
            },
            reinit=True,
        )
        # Disable DeepSpeed's built-in wandb to avoid duplicate logging
        ds_config["wandb"] = {"enabled": False}

    os.makedirs(out_cfg["dir"], exist_ok=True)
    os.makedirs(out_cfg["checkpoint_dir"], exist_ok=True)
    log_path = os.path.join(out_cfg["dir"], "train_log.jsonl")

    global_step = 0
    best_loss = float("inf")

    for epoch in range(train_cfg["epochs"]):
        if sampler is not None:
            sampler.set_epoch(epoch)

        model_engine.train()
        epoch_losses = []
        t0 = time.time()

        for batch in loader:
            model_dtype = next(model_engine.parameters()).dtype
            batch_dev = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    v = v.to(device)
                    if v.is_floating_point():
                        v = v.to(model_dtype)
                batch_dev[k] = v

            outputs = model_engine(batch_dev)

            # Masked recon loss -- pass masks so only masked+valid beats count
            loss_masked_d = l_masked(
                outputs["recon"], outputs["target_beats"],
                beat_mask_recon=outputs.get("beat_mask_recon"),
                beat_masks=outputs.get("beat_masks"),
                window_mask=outputs.get("window_mask"),
            )
            loss_order_d = l_order(
                outputs["predictions"], outputs["target_window_embeds"],
                outputs["window_mask"], outputs["quality_mask"],
            )
            loss_reg_d = l_reg(outputs["window_embeds"], outputs["window_mask"])

            total_loss = (w_cfg["masked"] * loss_masked_d["l_masked"] +
                          w_cfg["order"] * loss_order_d["l_order"] +
                          w_cfg["var"] * loss_reg_d["l_var"] +
                          w_cfg["cov"] * loss_reg_d["l_cov"])

            # Day-level loss: predict target window mean from day embedding
            if "day_prediction" in outputs and "target_window_embeds" in outputs:
                with torch.no_grad():
                    wm = outputs["window_mask"].unsqueeze(-1).float()
                    target_mean = (outputs["target_window_embeds"] * wm).sum(1) / wm.sum(1).clamp(min=1)
                l_day = F.smooth_l1_loss(outputs["day_prediction"], target_mean)
                total_loss = total_loss + w_cfg.get("day", 0.5) * l_day
            else:
                l_day = torch.tensor(0.0)

            # DeepSpeed backward + step
            model_engine.backward(total_loss)
            model_engine.step()

            # EMA update on the underlying module
            model_engine.module.update_ema()

            global_step += 1
            epoch_losses.append(total_loss.item())

            if global_step % 10 == 0 and is_main_process():
                lr_current = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]["lr"]
                log_entry = {
                    "step": global_step, "epoch": epoch,
                    "loss": total_loss.item(),
                    "l_masked": loss_masked_d["l_masked"].item(),
                    "l_order": loss_order_d["l_order"].item(),
                    "l_var": loss_reg_d["l_var"].item(),
                    "l_cov": loss_reg_d["l_cov"].item(),
                    "l_day": l_day.item() if hasattr(l_day, "item") else 0.0,
                    "lr": lr_current,
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

                # wandb logging
                if HAS_WANDB:
                    wandb.log(log_entry, step=global_step)

        epoch_loss = np.mean(epoch_losses)
        elapsed = time.time() - t0

        if is_main_process():
            print(f"Epoch {epoch+1}/{train_cfg['epochs']} | Loss: {epoch_loss:.4f} | Time: {elapsed:.1f}s")
            if HAS_WANDB:
                wandb.log({"epoch": epoch + 1, "epoch_loss": epoch_loss,
                           "epoch_time_s": elapsed}, step=global_step)

        # Save best checkpoint (rank 0)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            if is_main_process():
                ckpt_path = os.path.join(out_cfg["checkpoint_dir"], "best.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model_engine.module.state_dict(),
                    "loss": best_loss, "config": cfg,
                }, ckpt_path)
                print(f"  Saved best checkpoint: {ckpt_path}")

        # DeepSpeed checkpoint (all ranks)
        model_engine.save_checkpoint(out_cfg["checkpoint_dir"], tag=f"epoch_{epoch}")

    # Final checkpoint
    if is_main_process():
        torch.save({
            "epoch": train_cfg["epochs"] - 1,
            "model_state_dict": model_engine.module.state_dict(),
            "loss": epoch_loss, "config": cfg,
        }, os.path.join(out_cfg["checkpoint_dir"], "final.pt"))
        print("Training complete!")
        if HAS_WANDB:
            wandb.finish()


if __name__ == "__main__":
    main()
