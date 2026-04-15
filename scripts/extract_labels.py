#!/usr/bin/env python3
"""Extract diagnostic labels from Holter CSV report text.

R007: NLP label extraction from 结论 (conclusion) field.

Supports two extraction methods:
  --method regex  (default) Fast rule-based regex matching
  --method llm    Lightweight LLM classification (GPU recommended)

Usage:
    # Regex (fast, no GPU needed)
    python scripts/extract_labels.py --data_dir data/DMS --output labels.json

    # LLM (more accurate, needs GPU + transformers)
    python scripts/extract_labels.py --data_dir data/DMS --output labels_llm.json \
        --method llm --model Qwen/Qwen2.5-1.5B-Instruct
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.data.loader import scan_data_dir, load_patient
from src.nlp.report_parser import parse_patient_labels, LABEL_NAMES


def _get_conclusion_text(csv_meta):
    """Extract conclusion text from patient metadata."""
    conclusion_keys = ["conclusion", "\u7ed3\u8bba", "结论"]
    for key in conclusion_keys:
        if key in csv_meta:
            return str(csv_meta[key])
    for key, val in csv_meta.items():
        if "结论" in str(key) or "conclusion" in str(key).lower():
            return str(val)
    return ""


def main():
    parser = argparse.ArgumentParser(description="Extract NLP labels from Holter reports")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="labels.json")
    parser.add_argument("--method", type=str, choices=["regex", "llm"], default="regex",
                        help="Label extraction method: regex (fast) or llm (accurate, GPU)")
    parser.add_argument("--model", type=str, default=None,
                        help="HuggingFace model name for LLM method (default: Qwen/Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for LLM inference (default: auto)")
    args = parser.parse_args()

    records = scan_data_dir(args.data_dir)
    print(f"Found {len(records)} patient records")
    print(f"Method: {args.method}")

    # Initialize LLM extractor if needed
    llm_extractor = None
    if args.method == "llm":
        from src.nlp.llm_label_extractor import LLMLabelExtractor
        llm_extractor = LLMLabelExtractor(
            model_name=args.model,
            device=args.device,
        )

    results = []
    label_counts = {ln: 0 for ln in LABEL_NAMES}
    total = 0

    for rec in tqdm(records, desc="Label extraction"):
        try:
            patient = load_patient(rec)
            pid = patient.get("pid", rec["stem"])
            # Guard against NaN pid from CSV parsing
            if pid is None or (isinstance(pid, float) and np.isnan(pid)):
                pid = rec["stem"]

            if args.method == "regex":
                labels = parse_patient_labels(patient)
                binary = labels.get("binary", {})
                numeric = labels.get("numeric", {})
                raw_text = _get_conclusion_text(patient)
            else:
                raw_text = _get_conclusion_text(patient)
                binary = llm_extractor.extract(raw_text)
                # Still extract numeric via regex (LLM not needed for numbers)
                from src.nlp.report_parser import extract_numeric_from_text
                numeric = extract_numeric_from_text(raw_text)

            results.append({
                "pid": pid,
                "raw_text": raw_text,
                "binary": binary,
                "numeric": numeric,
            })

            total += 1
            for ln in LABEL_NAMES:
                if binary.get(ln, 0):
                    label_counts[ln] += 1

        except Exception as e:
            print(f"  ERROR {rec['stem']}: {e}")

    output = {
        "method": args.method,
        "model": args.model if args.method == "llm" else "regex",
        "label_names": LABEL_NAMES,
        "total_patients": total,
        "label_stats": {k: f"{v}/{total}" for k, v in label_counts.items()},
        "prevalence": {k: v / max(total, 1) for k, v in label_counts.items()},
        "patients": results,
    }

    # Resolve output path
    output_path = args.output
    if os.path.isdir(output_path) or output_path.endswith("/"):
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, f"labels_{args.method}.json")
    else:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # --- Save JSON (human-readable, with raw text and stats) ---
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # --- Save npz (for downstream: linear_probe.py, dataset.py) ---
    pids = [r["pid"] for r in results]
    label_matrix = np.array(
        [[r["binary"].get(ln, 0) for ln in LABEL_NAMES] for r in results],
        dtype=np.int32,
    )
    npz_path = output_path.replace(".json", ".npz")
    if npz_path == output_path:
        npz_path = output_path + ".npz"
    np.savez(
        npz_path,
        labels=label_matrix,
        label_names=np.array(LABEL_NAMES),
        patient_ids=np.array(pids),
    )

    print(f"\nExtracted labels for {total} patients ({args.method})")
    print("Label prevalence:")
    for ln in LABEL_NAMES:
        prev = label_counts[ln] / max(total, 1)
        print(f"  {ln}: {label_counts[ln]}/{total} ({prev:.1%})")
    print(f"JSON: {output_path}")
    print(f"NPZ:  {npz_path}  (shape: {label_matrix.shape})")


if __name__ == "__main__":
    main()
