"""Preprocess raw Holter data into .npz files for training."""
import os, sys, argparse, yaml, json, time
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.loader import scan_data_dir, load_patient
from src.data.beat_segmentation import detect_r_peaks_multichannel, segment_beats, beats_to_windows
from src.data.preprocessing import prepare_patient_windows
from src.nlp.report_parser import parse_patient_labels


def _parse_start_time_of_day(meta):
    """Extract recording start time-of-day in seconds from CSV metadata."""
    start_dt = str(meta.get("start_datetime", ""))
    try:
        time_part = start_dt.strip().split()[-1]  # "07:55"
        h, m = time_part.split(":")
        return int(h) * 3600 + int(m) * 60
    except Exception:
        return 0.0


def preprocess_patient(record, out_dir, max_beats=300, window_sec=300):
    """Process one patient record into .npz."""
    try:
        meta = load_patient(record)
        ecg = meta["ecg"]
    except Exception as e:
        return {"pid": record["stem"], "status": "skip", "reason": str(e)}

    r_peaks = detect_r_peaks_multichannel(ecg)
    if len(r_peaks) < 50:
        return {"pid": record["stem"], "status": "skip",
                "reason": f"too few R-peaks ({len(r_peaks)})"}

    beats, peak_indices = segment_beats(ecg, r_peaks)
    if len(beats) < 20:
        return {"pid": record["stem"], "status": "skip",
                "reason": f"too few beats ({len(beats)})"}

    windows = beats_to_windows(beats, peak_indices, window_sec=window_sec)
    if len(windows) == 0:
        return {"pid": record["stem"], "status": "skip",
                "reason": "no valid windows"}

    # Critical: pass actual recording start time for correct circadian encoding
    start_time_sec = _parse_start_time_of_day(meta)
    prepared = prepare_patient_windows(
        windows,
        max_beats_per_window=max_beats,
        start_time_sec=start_time_sec,
    )

    patient_id = record["stem"]
    labels_data = parse_patient_labels(meta)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{patient_id}.npz")
    np.savez_compressed(
        out_path,
        beat_tensors=prepared["beat_tensors"],
        masks=prepared["masks"],
        time_encodings=prepared["time_encodings"],
        quality_mask=prepared["quality_mask"],
        patient_id=patient_id,
        labels_binary=labels_data["binary"],
        labels_numeric=labels_data["numeric"],
    )
    return {"pid": patient_id, "status": "ok",
            "n_windows": len(windows), "n_beats": len(beats),
            "start_tod": start_time_sec}


def _worker(record, out_dir, max_beats, window_sec):
    return preprocess_patient(record, out_dir, max_beats, window_sec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--workers", type=int, default=8,
                        help="parallel workers (0=sequential)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = args.data_dir or cfg["data"]["data_dir"]
    out_dir = cfg["data"]["processed_dir"]
    max_beats = cfg["data"].get("max_beats_per_window", 300)
    window_sec = cfg["data"].get("window_sec", 300)

    patients = scan_data_dir(data_dir)
    print(f"Found {len(patients)} patients in {data_dir}")
    print(f"Output: {out_dir}  |  Workers: {args.workers}")

    t0 = time.time()
    fn = partial(_worker, out_dir=out_dir, max_beats=max_beats, window_sec=window_sec)

    if args.workers > 0:
        with Pool(min(args.workers, cpu_count())) as pool:
            results = pool.map(fn, patients)
    else:
        results = [fn(r) for r in patients]

    dt = time.time() - t0
    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skip"]

    print(f"\nDone in {dt:.0f}s: {len(ok)} OK, {len(skipped)} skipped")
    if skipped:
        print(f"Skipped reasons:")
        from collections import Counter
        for reason, cnt in Counter(r["reason"] for r in skipped).most_common():
            print(f"  {cnt}x {reason}")

    # Save preprocessing report
    os.makedirs(out_dir, exist_ok=True)
    report = {
        "total": len(patients), "ok": len(ok), "skipped": len(skipped),
        "time_sec": round(dt, 1),
        "skipped_details": skipped,
        "window_stats": {
            "mean": round(np.mean([r["n_windows"] for r in ok]), 1) if ok else 0,
            "min": min((r["n_windows"] for r in ok), default=0),
            "max": max((r["n_windows"] for r in ok), default=0),
        },
    }
    report_path = os.path.join(out_dir, "preprocess_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
