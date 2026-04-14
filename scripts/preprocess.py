"""Preprocess raw Holter data into .npz files for training."""
import os, sys, argparse, yaml
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.loader import scan_data_dir, load_patient
from src.data.beat_segmentation import detect_r_peaks_multichannel, segment_beats, beats_to_windows
from src.data.preprocessing import prepare_patient_windows
from src.nlp.report_parser import parse_patient_labels


def preprocess_patient(record, out_dir, max_beats=300, window_sec=300):
    """Process one patient record into .npz."""
    try:
        meta = load_patient(record)
        ecg = meta["ecg"]
    except Exception as e:
        print(f"  SKIP {record['stem']}: {e}")
        return False

    r_peaks = detect_r_peaks_multichannel(ecg)
    if len(r_peaks) < 50:
        print(f"  SKIP {record['stem']}: too few R-peaks ({len(r_peaks)})")
        return False

    beats, peak_indices = segment_beats(ecg, r_peaks)
    if len(beats) < 20:
        print(f"  SKIP {record['stem']}: too few beats ({len(beats)})")
        return False

    windows = beats_to_windows(beats, peak_indices, window_sec=window_sec)
    if len(windows) == 0:
        print(f"  SKIP {record['stem']}: no valid windows")
        return False

    prepared = prepare_patient_windows(windows, max_beats_per_window=max_beats)

    # Extract patient ID from record stem
    patient_id = record["stem"]

    # Parse labels from CSV metadata
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
    print(f"  OK {patient_id}: {len(windows)} windows -> {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_dir", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = args.data_dir or cfg["data"]["data_dir"]
    out_dir = cfg["data"]["processed_dir"]
    max_beats = cfg["data"].get("max_beats_per_window", 300)
    window_sec = cfg["data"].get("window_sec", 300)

    patients = scan_data_dir(data_dir)
    print(f"Found {len(patients)} patients in {data_dir}")

    success = 0
    for record in patients:
        if preprocess_patient(record, out_dir, max_beats, window_sec):
            success += 1

    print(f"\nPreprocessing complete: {success}/{len(patients)} patients")


if __name__ == "__main__":
    main()
