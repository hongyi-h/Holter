"""Extract NLP labels from CSV conclusions."""
import os, sys, argparse, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.loader import scan_data_dir, load_patient
from src.nlp.report_parser import parse_patient_labels, LABEL_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    patients = scan_data_dir(args.data_dir)
    print(f"Found {len(patients)} patients")

    all_labels = {}
    label_counts = {name: 0 for name in LABEL_NAMES}

    for record in patients:
        pid = record["stem"]
        try:
            meta = load_patient(record)
            labels = parse_patient_labels(meta)
            all_labels[pid] = labels
            for name in LABEL_NAMES:
                label_counts[name] += labels["binary"].get(name, 0)
        except Exception as e:
            print(f"  {pid}: SKIP - {e}")

    print(f"\nLabel prevalence ({len(all_labels)} patients):")
    for name in LABEL_NAMES:
        pct = label_counts[name] / max(len(all_labels), 1) * 100
        print(f"  {name}: {label_counts[name]} ({pct:.1f}%)")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Save as npz for probe dataset
    patient_ids = sorted(all_labels.keys())
    label_matrix = np.array([[all_labels[pid]["binary"].get(name, 0) for name in LABEL_NAMES]
                             for pid in patient_ids])
    np.savez(args.output, labels=label_matrix, patient_ids=patient_ids, label_names=LABEL_NAMES)
    print(f"Saved labels to {args.output}")


if __name__ == "__main__":
    main()
