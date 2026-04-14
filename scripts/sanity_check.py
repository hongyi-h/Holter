"""Sanity check: verify data pipeline, model, losses, and metrics."""
import os, sys, argparse, yaml, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sanity_data_pipeline(data_dir):
    from src.data.loader import scan_data_dir, load_patient
    from src.data.beat_segmentation import detect_r_peaks_multichannel, segment_beats, beats_to_windows
    from src.data.preprocessing import prepare_patient_windows

    patients = scan_data_dir(data_dir)
    assert len(patients) > 0, f"No patients found in {data_dir}"
    print(f"  Found {len(patients)} patient(s)")

    meta = load_patient(patients[0])
    ecg = meta["ecg"]
    print(f"  ECG shape: {ecg.shape}, dtype: {ecg.dtype}")
    assert ecg.ndim == 2 and ecg.shape[1] == 3

    from src.data.beat_segmentation import detect_r_peaks_multichannel, segment_beats, beats_to_windows
    r_peaks = detect_r_peaks_multichannel(ecg)
    print(f"  R-peaks detected: {len(r_peaks)}")
    assert len(r_peaks) > 50

    beats, peak_indices = segment_beats(ecg, r_peaks)
    print(f"  Beats segmented: {beats.shape}")

    windows = beats_to_windows(beats, peak_indices)
    print(f"  Windows created: {len(windows)}")
    assert len(windows) > 0

    prepared = prepare_patient_windows(windows, max_beats_per_window=300)
    bt = prepared["beat_tensors"]
    print(f"  Prepared tensors: {bt.shape}")

    return {"ecg": ecg, "meta": meta, "r_peaks": r_peaks, "beats": beats,
            "windows": windows, "prepared": prepared}


def sanity_hrv(processed):
    from src.metrics.hrv import extract_all_hrv
    hrv = extract_all_hrv(processed["r_peaks"])
    assert "SDNN" in hrv and hrv["SDNN"] > 0
    print(f"  SDNN={hrv['SDNN']:.1f} RMSSD={hrv['RMSSD']:.1f} LF/HF={hrv['LF_HF_ratio']:.2f}")
    return hrv


def sanity_cer(processed):
    from src.metrics.cer import compute_cer
    n_windows = len(processed["windows"])
    fake_embeds = np.random.randn(n_windows, 256).astype(np.float32)
    quality_mask = np.ones(n_windows, dtype=bool)
    result = compute_cer(fake_embeds, quality_mask)
    print(f"  CER={result['CER']:.4f} H1={result['H1']:.4f} F={result['F']:.4f} status={result['status']}")
    return result


def sanity_nlp(processed):
    from src.nlp.report_parser import parse_patient_labels
    labels = parse_patient_labels(processed["meta"])
    print(f"  Labels: {labels['binary']}")
    print(f"  Numeric: {labels['numeric']}")
    return labels


def sanity_model_forward(processed, cfg):
    from src.models.holter_ssl import HolterSSL
    from src.losses.masked_recon import MaskedReconLoss
    from src.losses.order_prediction import OrderPredictionLoss
    from src.losses.regularization import AntiCollapseLoss

    model = HolterSSL(cfg["model"])
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {total_params:,} total, {trainable:,} trainable")

    bt = torch.from_numpy(processed["prepared"]["beat_tensors"]).float()
    bm = torch.from_numpy(processed["prepared"]["masks"])
    te = torch.from_numpy(processed["prepared"]["time_encodings"])
    qm = torch.from_numpy(processed["prepared"]["quality_mask"])

    max_w = min(6, bt.shape[0])
    batch = {
        "beat_tensors": bt[:max_w].unsqueeze(0),
        "beat_masks": bm[:max_w].unsqueeze(0),
        "time_encodings": te[:max_w].unsqueeze(0),
        "quality_mask": qm[:max_w].unsqueeze(0),
        "window_mask": torch.ones(1, max_w, dtype=torch.bool),
    }

    # --- Test TRAINING path ---
    model.train()
    outputs_train = model(batch)
    print(f"  [train] output keys: {sorted(outputs_train.keys())}")
    print(f"  [train] window_embeds: {outputs_train['window_embeds'].shape}")
    print(f"  [train] day_embed: {outputs_train['day_embed'].shape}")
    assert "recon" in outputs_train, "Training path must return recon"
    assert "target_beats" in outputs_train, "Training path must return target_beats"
    assert "beat_mask_recon" in outputs_train, "Training path must return beat_mask_recon"
    assert "day_prediction" in outputs_train, "Training path must return day_prediction"

    # Compute losses (training path)
    loss_cfg = cfg["loss"]
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

    loss_masked_d = l_masked(
        outputs_train["recon"], outputs_train["target_beats"],
        beat_mask_recon=outputs_train.get("beat_mask_recon"),
        beat_masks=outputs_train.get("beat_masks"),
        window_mask=outputs_train.get("window_mask"),
    )
    loss_order_d = l_order(
        outputs_train["predictions"], outputs_train["target_window_embeds"],
        outputs_train["window_mask"], outputs_train["quality_mask"],
    )
    loss_reg_d = l_reg(outputs_train["window_embeds"], outputs_train["window_mask"])

    import torch.nn.functional as F
    wm = outputs_train["window_mask"].unsqueeze(-1).float()
    target_mean = (outputs_train["target_window_embeds"] * wm).sum(1) / wm.sum(1).clamp(min=1)
    l_day = F.smooth_l1_loss(outputs_train["day_prediction"], target_mean)

    w_cfg = loss_cfg["weights"]
    total = (w_cfg["masked"] * loss_masked_d["l_masked"] +
             w_cfg["order"] * loss_order_d["l_order"] +
             w_cfg["var"] * loss_reg_d["l_var"] +
             w_cfg["cov"] * loss_reg_d["l_cov"] +
             w_cfg.get("day", 0.5) * l_day)

    print(f"  L_masked={loss_masked_d['l_masked']:.4f}, L_order={loss_order_d['l_order']:.4f}, "
          f"L_var={loss_reg_d['l_var']:.4f}, L_cov={loss_reg_d['l_cov']:.4f}, L_day={l_day:.4f}")
    print(f"  Total loss={total.item():.4f}")

    # Backprop test
    total.backward()
    day_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.day_encoder.parameters() if p.requires_grad)
    print(f"  DayEncoder receives gradient: {day_grad}")

    model.update_ema()
    print("  EMA update OK")

    # --- Test EVAL path ---
    model.eval()
    with torch.no_grad():
        outputs_eval = model(batch)
    print(f"  [eval] output keys: {sorted(outputs_eval.keys())}")
    assert "recon" not in outputs_eval, "Eval path should NOT return recon"
    assert "window_embeds" in outputs_eval
    print(f"  [eval] window_embeds: {outputs_eval['window_embeds'].shape}")
    print(f"  [eval] day_embed: {outputs_eval['day_embed'].shape}")

    return {"total_loss": total.item(), "params": total_params}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sanity.yaml")
    parser.add_argument("--data_dir", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = args.data_dir or cfg["data"]["data_dir"]

    results = {}
    print("\n=== 1. Data Pipeline ===")
    processed = sanity_data_pipeline(data_dir)
    results["data_pipeline"] = "PASS"

    print("\n=== 2. HRV Features ===")
    hrv = sanity_hrv(processed)
    results["hrv"] = "PASS"

    print("\n=== 3. CER Metric ===")
    cer = sanity_cer(processed)
    results["cer"] = "PASS"

    print("\n=== 4. NLP Report Parser ===")
    nlp = sanity_nlp(processed)
    results["nlp"] = "PASS"

    print("\n=== 5. Model Forward Pass ===")
    model_result = sanity_model_forward(processed, cfg)
    results["model"] = "PASS"

    print("\n" + "=" * 50)
    print("ALL SANITY CHECKS PASSED")
    print("=" * 50)

    out_dir = cfg.get("output", {}).get("dir", "results/sanity")
    os.makedirs(out_dir, exist_ok=True)
    results["model_params"] = model_result["params"]
    results["total_loss"] = model_result["total_loss"]
    results["hrv_sdnn"] = hrv["SDNN"]
    results["cer_value"] = cer["CER"]
    with open(os.path.join(out_dir, "sanity_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_dir}/sanity_results.json")


if __name__ == "__main__":
    main()
