"""Downstream fine-tuning / evaluation for HolterFM (M3: R030-R034)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast

from src.models.holter_fm import HolterFM
from src.data.holter_dataset import HolterPretrainDataset, collate_holter
from src.data.holter_record import HolterRecord
from src.data.report_concepts import ReportConceptExtractor
from src.evaluation.downstream_heads import (
    BeatClassificationHead, PVCBurdenHead, ReportConceptHead, VentricularEventHead,
)


def load_pretrained(checkpoint: str, device: torch.device) -> HolterFM:
    model = HolterFM()
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    return model


def freeze_backbone(model: HolterFM):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_top_episode_layers(model: HolterFM, n: int = 2):
    for layer in model.episode_encoder.layers[-n:]:
        for p in layer.parameters():
            p.requires_grad = True
    for p in model.episode_encoder.norm.parameters():
        p.requires_grad = True


def split_records(data_dir: str, seed: int = 42):
    records = HolterRecord.discover(data_dir)
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(records), generator=rng).tolist()
    n = len(records)
    n_train, n_val = int(n * 0.7), int(n * 0.1)
    return (
        [records[i] for i in idx[:n_train]],
        [records[i] for i in idx[n_train:n_train + n_val]],
        [records[i] for i in idx[n_train + n_val:]],
    )


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_beat_metrics(all_preds: list, all_labels: list) -> dict:
    from collections import Counter
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    classes = [0, 1, 2]
    names = ["N", "V", "F"]
    metrics = {}
    for c, name in zip(classes, names):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        metrics[f"f1_{name}"] = float(f1)
    metrics["macro_f1"] = np.mean([metrics[f"f1_{n}"] for n in names])
    metrics["accuracy"] = float((preds == labels).mean())
    return metrics


def compute_burden_metrics(all_preds: list, all_targets: list) -> dict:
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    mae = np.abs(preds[:, 0] - targets[:, 0]).mean()
    from scipy.stats import spearmanr
    rho, _ = spearmanr(preds[:, 0], targets[:, 0])
    ss_res = ((preds[:, 0] - targets[:, 0]) ** 2).sum()
    ss_tot = ((targets[:, 0] - targets[:, 0].mean()) ** 2).sum()
    r2 = 1 - ss_res / max(ss_tot, 1e-8)
    return {"mae_burden": float(mae), "spearman_rho": float(rho), "r2": float(r2)}


def compute_concept_metrics(all_logits: list, all_labels: list) -> dict:
    from sklearn.metrics import roc_auc_score, average_precision_score
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    probs = 1 / (1 + np.exp(-logits))
    n_concepts = labels.shape[1]
    aurocs, auprcs = [], []
    for i in range(n_concepts):
        if labels[:, i].sum() > 0 and labels[:, i].sum() < len(labels):
            aurocs.append(roc_auc_score(labels[:, i], probs[:, i]))
            auprcs.append(average_precision_score(labels[:, i], probs[:, i]))
    return {
        "macro_auroc": float(np.mean(aurocs)) if aurocs else 0.0,
        "macro_auprc": float(np.mean(auprcs)) if auprcs else 0.0,
        "n_evaluable_concepts": len(aurocs),
    }


# ── Task runners ─────────────────────────────────────────────────────────────

def run_beat_classification(
    checkpoint: str, data_dir: str, mode: str = "linear_probe",
    epochs: int = 20, lr: float = 1e-3, lr_backbone: float = 5e-5,
    seed: int = 42, device_str: str = "cuda",
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    train_rec, val_rec, test_rec = split_records(data_dir, seed)

    model = load_pretrained(checkpoint, device)
    freeze_backbone(model)
    if mode == "fine_tune":
        unfreeze_top_episode_layers(model, n=2)

    head = BeatClassificationHead().to(device)

    param_groups = [{"params": head.parameters(), "lr": lr}]
    if mode == "fine_tune":
        backbone_params = [p for p in model.parameters() if p.requires_grad]
        param_groups.append({"params": backbone_params, "lr": lr_backbone})
    optimizer = torch.optim.AdamW(param_groups)

    train_ds = HolterPretrainDataset(train_rec, augment=(mode == "fine_tune"))
    val_ds = HolterPretrainDataset(val_rec, augment=False)
    test_ds = HolterPretrainDataset(test_rec, augment=False)

    def make_loader(ds, shuffle=False):
        return DataLoader(ds, batch_size=1, shuffle=shuffle, collate_fn=collate_holter,
                          num_workers=2, pin_memory=True)

    train_loader = make_loader(train_ds, shuffle=True)
    val_loader = make_loader(val_ds)
    test_loader = make_loader(test_ds)

    best_val_f1 = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train() if mode == "fine_tune" else model.eval()
        head.train()
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad() if mode == "linear_probe" else torch.enable_grad():
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(batch)
            logits = head(out["beat_repr"].float())
            loss = head.compute_loss(logits, batch["beat_labels"], batch["valid_mask"])
            loss.backward()
            optimizer.step()

        # validate
        model.eval()
        head.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                out = model(batch)
                logits = head(out["beat_repr"].float())
                mask = batch["valid_mask"]
                preds = logits[mask].argmax(dim=-1).cpu().numpy()
                labels = batch["beat_labels"][mask].cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels)

        metrics = compute_beat_metrics(all_preds, all_labels)
        print(f"  Epoch {epoch}: val macro_f1={metrics['macro_f1']:.4f} "
              f"f1_V={metrics['f1_V']:.4f} f1_F={metrics['f1_F']:.4f}")

        if metrics["macro_f1"] > best_val_f1:
            best_val_f1 = metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}

    # test
    head.load_state_dict(best_state)
    head = head.to(device)
    model.eval()
    head.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            out = model(batch)
            logits = head(out["beat_repr"].float())
            mask = batch["valid_mask"]
            preds = logits[mask].argmax(dim=-1).cpu().numpy()
            labels = batch["beat_labels"][mask].cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels)

    test_metrics = compute_beat_metrics(all_preds, all_labels)
    test_metrics["best_val_f1"] = best_val_f1
    return test_metrics


def run_pvc_burden(
    checkpoint: str, data_dir: str, mode: str = "frozen",
    epochs: int = 100, lr: float = 1e-3, patience: int = 15,
    seed: int = 42, device_str: str = "cuda",
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    train_rec, val_rec, test_rec = split_records(data_dir, seed)

    model = load_pretrained(checkpoint, device)
    freeze_backbone(model)
    head = PVCBurdenHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)

    train_ds = HolterPretrainDataset(train_rec)
    val_ds = HolterPretrainDataset(val_rec)
    test_ds = HolterPretrainDataset(test_rec)

    def make_loader(ds, shuffle=False):
        return DataLoader(ds, batch_size=1, shuffle=shuffle, collate_fn=collate_holter,
                          num_workers=2, pin_memory=True)

    best_val_mae = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.eval()
        head.train()
        for batch in make_loader(train_ds, shuffle=True):
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                out = model(batch)
            pred = head(out["day_embed"].float(), out["episode_ctx"].float(), batch["n_episodes"])
            loss = head.compute_loss(pred, batch["day_stats"])
            loss.backward()
            optimizer.step()

        # validate
        head.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in make_loader(val_ds):
                batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                out = model(batch)
                pred = head(out["day_embed"].float(), out["episode_ctx"].float(), batch["n_episodes"])
                all_preds.append(pred.cpu().numpy())
                burden = batch["day_stats"][:, 7:8]
                count = torch.log1p(batch["day_stats"][:, 6:7])
                all_targets.append(torch.cat([burden, count], dim=-1).cpu().numpy())

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        val_mae = np.abs(preds[:, 0] - targets[:, 0]).mean()

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # test
    head.load_state_dict(best_state)
    head = head.to(device)
    head.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in make_loader(test_ds):
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            out = model(batch)
            pred = head(out["day_embed"].float(), out["episode_ctx"].float(), batch["n_episodes"])
            all_preds.append(pred.cpu().numpy())
            burden = batch["day_stats"][:, 7:8]
            count = torch.log1p(batch["day_stats"][:, 6:7])
            all_targets.append(torch.cat([burden, count], dim=-1).cpu().numpy())

    return compute_burden_metrics(all_preds, all_targets)


def run_report_concepts(
    checkpoint: str, data_dir: str, epochs: int = 30, lr: float = 5e-4,
    seed: int = 42, device_str: str = "cuda",
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    train_rec, val_rec, test_rec = split_records(data_dir, seed)

    extractor = ReportConceptExtractor()
    n_concepts = extractor.n_concepts

    model = load_pretrained(checkpoint, device)
    freeze_backbone(model)
    head = ReportConceptHead(n_concepts=n_concepts).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)

    train_ds = HolterPretrainDataset(train_rec, day_stats_from_report=True)
    val_ds = HolterPretrainDataset(val_rec, day_stats_from_report=True)
    test_ds = HolterPretrainDataset(test_rec, day_stats_from_report=True)

    def make_loader(ds, shuffle=False):
        return DataLoader(ds, batch_size=1, shuffle=shuffle, collate_fn=collate_holter,
                          num_workers=2, pin_memory=True)

    best_val_auroc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.eval()
        head.train()
        for batch in make_loader(train_ds, shuffle=True):
            if "concept_labels" not in batch:
                continue
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                out = model(batch)
            logits = head(out["day_embed"].float())
            loss = head.compute_loss(logits, batch["concept_labels"])
            loss.backward()
            optimizer.step()

        # validate
        head.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in make_loader(val_ds):
                if "concept_labels" not in batch:
                    continue
                batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                out = model(batch)
                logits = head(out["day_embed"].float())
                all_logits.append(logits.cpu().numpy())
                all_labels.append(batch["concept_labels"].cpu().numpy())

        if all_logits:
            metrics = compute_concept_metrics(all_logits, all_labels)
            if metrics["macro_auroc"] > best_val_auroc:
                best_val_auroc = metrics["macro_auroc"]
                best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}

    # test
    if best_state is None:
        return {"macro_auroc": 0.0, "note": "no concept labels available"}
    head.load_state_dict(best_state)
    head = head.to(device)
    head.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in make_loader(test_ds):
            if "concept_labels" not in batch:
                continue
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            out = model(batch)
            logits = head(out["day_embed"].float())
            all_logits.append(logits.cpu().numpy())
            all_labels.append(batch["concept_labels"].cpu().numpy())

    return compute_concept_metrics(all_logits, all_labels)


# ── Main ─────────────────────────────────────────────────────────────────────

TASK_RUNNERS = {
    "beat_classification": run_beat_classification,
    "pvc_burden": run_pvc_burden,
    "report_concepts": run_report_concepts,
}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="M3: Downstream evaluation")
    parser.add_argument("--task", type=str, required=True, choices=list(TASK_RUNNERS.keys()))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/DMS")
    parser.add_argument("--mode", type=str, default="linear_probe")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    runner = TASK_RUNNERS[args.task]
    kwargs = {
        "checkpoint": args.checkpoint,
        "data_dir": args.data_dir,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "device_str": args.device,
    }
    if args.task == "beat_classification":
        kwargs["mode"] = args.mode

    print(f"=== Task: {args.task} (mode={args.mode}, seed={args.seed}) ===")
    t0 = time.time()
    results = runner(**kwargs)
    dt = time.time() - t0

    results["task"] = args.task
    results["mode"] = args.mode
    results["seed"] = args.seed
    results["time_sec"] = round(dt, 1)

    print(f"\nResults ({dt:.0f}s):")
    for k, v in sorted(results.items()):
        print(f"  {k}: {v}")

    out_path = args.output or f"results_{args.task}_{args.mode}_s{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
