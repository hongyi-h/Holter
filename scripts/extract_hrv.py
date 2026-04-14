"""Batch HRV feature extraction from processed data."""
import os, sys, argparse, glob, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.loader import scan_data_dir, load_patient
from src.data.beat_segmentation import detect_r_peaks_multichannel
from src.metrics.hrv import extract_all_hrv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    patients = scan_data_dir(args.data_dir)
    print(f"Found {len(patients)} patients")

    all_hrv = {}
    for record in patients:
        pid = record["stem"]
        try:
            meta = load_patient(record)
            ecg = meta["ecg"]
            r_peaks = detect_r_peaks_multichannel(ecg)
            hrv = extract_all_hrv(r_peaks)
            all_hrv[pid] = hrv
            print(f"  {pid}: SDNN={hrv['SDNN']:.1f}")
        except Exception as e:
            print(f"  {pid}: SKIP - {e}")

    output_path = args.output
    if os.path.isdir(output_path) or output_path.endswith("/"):
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, "hrv_features.json")
    else:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_hrv, f, indent=2)
    print(f"Saved HRV features to {output_path}")


if __name__ == "__main__":
    main()
