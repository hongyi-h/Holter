"""Holter DAT + CSV data loading utilities."""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

SAMPLE_RATE = 128
N_CHANNELS = 3
BASELINE_VALUE = 128.0

CSV_COLUMNS = [
    "name", "pid", "sex", "age", "n_channels", "start_datetime",
    "duration_hhmm", "total_beats", "mean_hr", "min_hr", "max_hr",
    "pvc_count", "pvc_pct", "svpc_count", "svpc_pct", "conclusion",
]


def load_dat(dat_path: str, n_channels: int = N_CHANNELS) -> np.ndarray:
    raw = np.fromfile(dat_path, dtype=np.uint8)
    n_samples = len(raw) // n_channels
    raw = raw[: n_samples * n_channels]
    ecg = raw.reshape(-1, n_channels).astype(np.float32) - BASELINE_VALUE
    return ecg


def load_csv(csv_path: str) -> dict:
    df = pd.read_csv(csv_path, encoding="gbk", header=0, names=CSV_COLUMNS)
    if len(df) == 0:
        raise ValueError(f"Empty CSV: {csv_path}")
    row = df.iloc[0].to_dict()
    row["age"] = int(row["age"]) if pd.notna(row["age"]) else None
    row["pid"] = str(row["pid"]).strip() if pd.notna(row.get("pid")) else ""
    row["total_beats"] = int(row["total_beats"]) if pd.notna(row["total_beats"]) else 0
    row["mean_hr"] = int(row["mean_hr"]) if pd.notna(row["mean_hr"]) else None
    row["pvc_count"] = int(row["pvc_count"]) if pd.notna(row["pvc_count"]) else 0
    row["svpc_count"] = int(row["svpc_count"]) if pd.notna(row["svpc_count"]) else 0
    return row


def parse_duration(hhmm: str) -> float:
    parts = str(hhmm).split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60


def scan_data_dir(data_dir: str) -> list:
    data_dir = Path(data_dir)
    records = []
    for csv_path in sorted(data_dir.glob("*_HolterSummary.csv")):
        stem = csv_path.name.replace("_HolterSummary.csv", "")
        dat_path = data_dir / f"{stem}.dat"
        if not dat_path.exists():
            continue
        records.append({"csv_path": str(csv_path), "dat_path": str(dat_path), "stem": stem})
    return records


def load_patient(record: dict) -> dict:
    meta = load_csv(record["csv_path"])
    ecg = load_dat(record["dat_path"])
    duration_sec = parse_duration(meta["duration_hhmm"])
    meta["ecg"] = ecg
    meta["duration_sec"] = duration_sec
    meta["actual_samples"] = ecg.shape[0]
    meta["expected_samples"] = int(duration_sec * SAMPLE_RATE)
    meta["sample_rate"] = SAMPLE_RATE
    return meta
