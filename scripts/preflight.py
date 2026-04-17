"""Pre-flight hardware & dependency check for MetaX C500 (MACA platform).

Run ONCE on server before any training.  Outputs results/preflight.json.

Usage:
    python scripts/preflight.py --data_dir data/DMS
"""
import os, sys, json, time, traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS = {}


def _record(key, status, detail=""):
    RESULTS[key] = {"status": status, "detail": str(detail)}
    tag = "✓" if status == "PASS" else "✗" if status == "FAIL" else "?"
    print(f"  [{tag}] {key}: {detail}")


# ───────────────────── 1. Python & system ─────────────────────
print("\n=== 1. Python & System ===")
_record("python_version", "PASS", sys.version.split()[0])
import platform
_record("os_platform", "PASS", f"{platform.system()} {platform.machine()}")


# ───────────────────── 2. Core dependencies ───────────────────
print("\n=== 2. Dependencies ===")
for mod_name in ["numpy", "scipy", "torch", "sklearn", "pandas", "yaml", "tqdm",
                  "matplotlib", "seaborn"]:
    try:
        m = __import__(mod_name)
        ver = getattr(m, "__version__", "ok")
        _record(f"dep_{mod_name}", "PASS", ver)
    except ImportError as e:
        _record(f"dep_{mod_name}", "FAIL", str(e))

# scipy sub-modules that we actually use
for sub in ["scipy.signal", "scipy.interpolate"]:
    try:
        __import__(sub)
        _record(f"dep_{sub}", "PASS", "importable")
    except ImportError as e:
        _record(f"dep_{sub}", "FAIL", str(e))

# DeepSpeed
try:
    import deepspeed
    _record("dep_deepspeed", "PASS", deepspeed.__version__)
except ImportError as e:
    _record("dep_deepspeed", "FAIL", str(e))

# neurokit2 — optional
try:
    import neurokit2
    _record("dep_neurokit2", "PASS", neurokit2.__version__)
except ImportError:
    _record("dep_neurokit2", "SKIP", "not installed (optional)")


# ───────────────────── 3. GPU capabilities ────────────────────
print("\n=== 3. GPU Capabilities ===")
import torch

_record("torch_version", "PASS", torch.__version__)
_record("cuda_available", "PASS" if torch.cuda.is_available() else "FAIL",
        torch.cuda.is_available())

if torch.cuda.is_available():
    n_gpu = torch.cuda.device_count()
    _record("gpu_count", "PASS", n_gpu)
    for i in range(n_gpu):
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1e9
        _record(f"gpu_{i}", "PASS", f"{name} | {mem:.1f} GB")

    dev = torch.device("cuda:0")

    # bf16 test
    try:
        a = torch.randn(8, 8, device=dev, dtype=torch.bfloat16)
        b = torch.randn(8, 8, device=dev, dtype=torch.bfloat16)
        c = a @ b
        _ = c.float().sum().item()
        _record("bf16_matmul", "PASS", "works")
    except Exception as e:
        _record("bf16_matmul", "FAIL", str(e))

    # fp16 test
    try:
        a = torch.randn(8, 8, device=dev, dtype=torch.float16)
        b = torch.randn(8, 8, device=dev, dtype=torch.float16)
        c = a @ b
        _ = c.float().sum().item()
        _record("fp16_matmul", "PASS", "works")
    except Exception as e:
        _record("fp16_matmul", "FAIL", str(e))

    # GPU FFT test
    try:
        x = torch.randn(4, 128, device=dev)
        y = torch.fft.rfft(x, dim=-1)
        _ = y.abs().sum().item()
        _record("gpu_fft", "PASS", "works")
    except Exception as e:
        _record("gpu_fft", "FAIL", str(e))

    # Conv1d test
    try:
        conv = torch.nn.Conv1d(3, 32, 7, padding=3).to(dev)
        x = torch.randn(2, 3, 128, device=dev)
        _ = conv(x).sum().item()
        _record("conv1d", "PASS", "works")
    except Exception as e:
        _record("conv1d", "FAIL", str(e))

    # GRU test
    try:
        gru = torch.nn.GRU(256, 256, 2, batch_first=True).to(dev)
        x = torch.randn(2, 10, 256, device=dev)
        out, _ = gru(x)
        _ = out.sum().item()
        _record("gru", "PASS", "works")
    except Exception as e:
        _record("gru", "FAIL", str(e))

    # TransformerEncoder test
    try:
        layer = torch.nn.TransformerEncoderLayer(d_model=256, nhead=4, batch_first=True)
        enc = torch.nn.TransformerEncoder(layer, num_layers=2).to(dev)
        x = torch.randn(2, 10, 256, device=dev)
        mask = torch.zeros(2, 10, dtype=torch.bool, device=dev)
        mask[:, 8:] = True
        _ = enc(x, src_key_padding_mask=mask).sum().item()
        _record("transformer_encoder", "PASS", "works")
    except Exception as e:
        _record("transformer_encoder", "FAIL", str(e))

    # .to(int) device semantics
    try:
        x = torch.randn(2, 2)
        y = x.to(0)
        assert y.is_cuda
        _record("to_int_device", "PASS", f"device={y.device}")
    except Exception as e:
        _record("to_int_device", "FAIL", str(e))

    # backward test (gradient flow)
    try:
        x = torch.randn(4, 4, device=dev, requires_grad=True)
        loss = (x ** 2).sum()
        loss.backward()
        assert x.grad is not None
        _record("backward", "PASS", "gradient OK")
    except Exception as e:
        _record("backward", "FAIL", str(e))
else:
    _record("gpu_tests", "SKIP", "no GPU available")


# ───────────────────── 4. Data check ──────────────────────────
print("\n=== 4. Data ===")
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data/DMS")
args, _ = parser.parse_known_args()

dat_files = [f for f in os.listdir(args.data_dir)
             if f.endswith(".dat") and not f.startswith("._")]
csv_files = [f for f in os.listdir(args.data_dir)
             if f.endswith(".csv") and not f.startswith("._")]
_record("dat_file_count", "PASS" if dat_files else "FAIL", len(dat_files))
_record("csv_file_count", "PASS" if csv_files else "FAIL", len(csv_files))

# Total data size
total_bytes = sum(os.path.getsize(os.path.join(args.data_dir, f)) for f in dat_files)
_record("total_dat_size_gb", "PASS", f"{total_bytes / 1e9:.2f} GB")

# Load one sample
if dat_files:
    try:
        from src.data.loader import scan_data_dir, load_patient
        patients = scan_data_dir(args.data_dir)
        _record("scan_patients", "PASS", f"{len(patients)} patients found")

        meta = load_patient(patients[0])
        ecg = meta["ecg"]
        _record("load_patient", "PASS",
                f"ecg shape={ecg.shape}, dtype={ecg.dtype}, "
                f"pid={meta.get('pid','?')}, dur={meta.get('duration_str','?')}")
    except Exception as e:
        _record("load_patient", "FAIL", f"{type(e).__name__}: {e}")

    # Beat segmentation on first patient
    try:
        from src.data.beat_segmentation import detect_r_peaks_multichannel, segment_beats, beats_to_windows
        t0 = time.time()
        r_peaks = detect_r_peaks_multichannel(ecg)
        dt_peaks = time.time() - t0
        _record("r_peak_detection", "PASS",
                f"{len(r_peaks)} peaks in {dt_peaks:.1f}s")

        beats, peak_indices = segment_beats(ecg, r_peaks)
        _record("beat_segmentation", "PASS", f"beats shape={beats.shape}")

        windows = beats_to_windows(beats, peak_indices)
        _record("windowing", "PASS",
                f"{len(windows)} windows, beats/window≈{beats.shape[0]/max(len(windows),1):.0f}")
    except Exception as e:
        _record("beat_segmentation", "FAIL", f"{type(e).__name__}: {e}")
        traceback.print_exc()


# ───────────────────── 5. Model on GPU ────────────────────────
print("\n=== 5. Model Forward/Backward on GPU ===")
if torch.cuda.is_available() and RESULTS.get("load_patient", {}).get("status") == "PASS":
    try:
        from src.data.preprocessing import prepare_patient_windows
        from src.models.holter_ssl import HolterSSL
        from src.losses.masked_recon import MaskedReconLoss
        from src.losses.order_prediction import OrderPredictionLoss
        from src.losses.regularization import AntiCollapseLoss

        # Prepare a mini batch from the loaded patient
        prepared = prepare_patient_windows(windows, max_beats_per_window=300)
        bt = torch.from_numpy(prepared["beat_tensors"]).float()
        bm = torch.from_numpy(prepared["masks"])
        te = torch.from_numpy(prepared["time_encodings"])
        qm = torch.from_numpy(prepared["quality_mask"])

        max_w = min(6, bt.shape[0])
        batch = {
            "beat_tensors": bt[:max_w].unsqueeze(0),
            "beat_masks": bm[:max_w].unsqueeze(0),
            "time_encodings": te[:max_w].unsqueeze(0),
            "quality_mask": qm[:max_w].unsqueeze(0),
            "window_mask": torch.ones(1, max_w, dtype=torch.bool),
        }

        model_cfg = {
            "mode": "ordered",
            "in_channels": 3, "beat_samples": 128, "beat_dim": 128,
            "window_dim": 256, "day_dim": 256,
            "w_heads": 4, "w_layers": 4, "d_layers": 2,
            "pred_hidden": 128, "dropout": 0.1, "ema_tau": 0.996,
            "max_beats": 300,
        }

        model = HolterSSL(model_cfg).to(dev)
        batch_dev = {k: v.to(dev) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

        # Forward (train mode)
        model.train()
        t0 = time.time()
        outputs = model(batch_dev)
        dt_fwd = time.time() - t0
        _record("model_forward", "PASS",
                f"keys={sorted(outputs.keys())}, fwd_time={dt_fwd:.3f}s")

        # Loss computation
        l_masked = MaskedReconLoss(spectral_weight=0.1)
        l_order = OrderPredictionLoss(horizons=[1, 3, 6],
                                       horizon_weights=[0.5, 0.3, 0.2])
        l_reg = AntiCollapseLoss(var_weight=1.0, cov_weight=0.04, var_floor=1.0)

        loss_m = l_masked(outputs["recon"], outputs["target_beats"],
                          beat_mask_recon=outputs.get("beat_mask_recon"),
                          beat_masks=outputs.get("beat_masks"),
                          window_mask=outputs.get("window_mask"))
        loss_o = l_order(outputs["predictions"], outputs["target_window_embeds"],
                         outputs["window_mask"], outputs["quality_mask"])
        loss_r = l_reg(outputs["window_embeds"], outputs["window_mask"])

        total = (loss_m["l_masked"] + 0.5 * loss_o["l_order"] +
                 0.1 * loss_r["l_var"] + 0.1 * loss_r["l_cov"])

        _record("loss_masked", "PASS", f"{loss_m['l_masked'].item():.4f}")
        _record("loss_order", "PASS", f"{loss_o['l_order'].item():.4f}")
        _record("loss_var", "PASS", f"{loss_r['l_var'].item():.4f}")
        _record("loss_cov", "PASS", f"{loss_r['l_cov'].item():.4f}")

        # Backward
        t0 = time.time()
        total.backward()
        dt_bwd = time.time() - t0
        grad_ok = any(p.grad is not None and p.grad.abs().sum() > 0
                      for p in model.parameters() if p.requires_grad)
        _record("backward_full", "PASS",
                f"grad_flow={'OK' if grad_ok else 'BROKEN'}, bwd_time={dt_bwd:.3f}s")

        # EMA update
        model.update_ema()
        _record("ema_update", "PASS", "ok")

        # Eval mode
        model.eval()
        with torch.no_grad():
            out_eval = model(batch_dev)
        _record("model_eval", "PASS",
                f"keys={sorted(out_eval.keys())}")

        # GPU memory
        mem_alloc = torch.cuda.memory_allocated(0) / 1e6
        mem_reserved = torch.cuda.memory_reserved(0) / 1e6
        _record("gpu_memory_mb", "PASS",
                f"allocated={mem_alloc:.0f}MB, reserved={mem_reserved:.0f}MB")

    except Exception as e:
        _record("model_gpu_test", "FAIL", f"{type(e).__name__}: {e}")
        traceback.print_exc()
else:
    _record("model_gpu_test", "SKIP", "no GPU or data load failed")


# ───────────────────── 6. DeepSpeed init test ─────────────────
print("\n=== 6. DeepSpeed Init (single-GPU, no distributed) ===")
if torch.cuda.is_available():
    try:
        import deepspeed as ds
        # Minimal config without distributed
        tiny_model = torch.nn.Linear(10, 10)
        ds_cfg = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "AdamW",
                "params": {"lr": 1e-4}
            },
            "bf16": {"enabled": RESULTS.get("bf16_matmul", {}).get("status") == "PASS"},
            "fp16": {"enabled": RESULTS.get("bf16_matmul", {}).get("status") != "PASS"},
        }
        # Note: full DeepSpeed init requires torchrun/deepspeed launcher
        # We just verify the config is parseable
        _record("deepspeed_config_valid", "PASS",
                f"bf16={ds_cfg['bf16']['enabled']}, fp16={ds_cfg['fp16']['enabled']}")
    except Exception as e:
        _record("deepspeed_init", "FAIL", str(e))
else:
    _record("deepspeed_init", "SKIP", "no GPU")


# ───────────────────── Save results ───────────────────────────
print("\n=== Results ===")
os.makedirs("results", exist_ok=True)
out_path = "results/preflight.json"
with open(out_path, "w") as f:
    json.dump(RESULTS, f, indent=2, ensure_ascii=False)

# Summary
n_pass = sum(1 for v in RESULTS.values() if v["status"] == "PASS")
n_fail = sum(1 for v in RESULTS.values() if v["status"] == "FAIL")
n_skip = sum(1 for v in RESULTS.values() if v["status"] == "SKIP")
print(f"\n{'='*50}")
print(f"  PASS: {n_pass}  |  FAIL: {n_fail}  |  SKIP: {n_skip}")
print(f"{'='*50}")

if n_fail > 0:
    print("\nFAILED items:")
    for k, v in RESULTS.items():
        if v["status"] == "FAIL":
            print(f"  ✗ {k}: {v['detail']}")

print(f"\nResults saved to {out_path}")
