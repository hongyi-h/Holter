#!/usr/bin/env python3
"""Audit DMS Holter files for beat-summary-report consistency.

This script answers the first design question for this project:

    Are RPointProperty, HolterSummary.csv, and the Chinese conclusion text
    internally consistent often enough to be treated as a report compiler, or
    inconsistent often enough to justify a quality-control study?

It intentionally does not train a model. It reconstructs deterministic
quantities from machine beat annotations and compares them with CSV fields and
free-text report statements.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


SAMPLE_RATE = 128
N_CHANNELS = 3
SUMMARY_SUFFIX = "_HolterSummary.csv"
RPOINT_SUFFIX = "_RPointProperty.txt"

CSV_COLUMNS = [
    "name",
    "pid",
    "sex",
    "age",
    "n_channels",
    "start_datetime",
    "duration_hhmm",
    "total_beats",
    "mean_hr",
    "min_hr",
    "max_hr",
    "pvc_count",
    "pvc_pct",
    "svpc_count",
    "svpc_pct",
    "conclusion",
]


@dataclass
class RecordPaths:
    stem: str
    csv_path: Path
    dat_path: Path
    rpoint_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit DMS Holter RPointProperty, summary CSV, and report text."
    )
    parser.add_argument("--data_dir", required=True, help="Directory containing DMS files.")
    parser.add_argument(
        "--output_dir",
        default="reports/data_audit",
        help="Directory for audit_records.csv, audit_mismatches.csv, and audit_summary.json.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan data_dir for *_HolterSummary.csv files.",
    )
    parser.add_argument(
        "--pvc_labels",
        default="V",
        help="Comma-separated beat labels counted as PVC. Default: V.",
    )
    parser.add_argument(
        "--svpc_labels",
        default="S",
        help=(
            "Comma-separated beat labels counted as SVPC/PAC. Default: S. "
            "Do not add A/AE/AP unless their DMS semantics are verified."
        ),
    )
    parser.add_argument(
        "--exclude_total_labels",
        default="F,FS",
        help="Comma-separated beat labels excluded from total beat count. Default: F,FS.",
    )
    parser.add_argument(
        "--pct_tolerance",
        type=float,
        default=0.02,
        help="Allowed absolute difference in percentage points for burden fields.",
    )
    parser.add_argument(
        "--hr_tolerance",
        type=float,
        default=1.0,
        help="Allowed absolute difference in bpm for heart-rate comparisons.",
    )
    parser.add_argument(
        "--rr_tolerance",
        type=float,
        default=0.02,
        help="Allowed absolute difference in seconds for longest R-R comparison.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for quick tests.",
    )
    return parser.parse_args()


def split_label_set(text: str) -> set[str]:
    return {x.strip() for x in text.split(",") if x.strip()}


def scan_records(data_dir: Path, recursive: bool) -> list[RecordPaths]:
    pattern = f"**/*{SUMMARY_SUFFIX}" if recursive else f"*{SUMMARY_SUFFIX}"
    records = []
    for csv_path in sorted(data_dir.glob(pattern)):
        if csv_path.name.startswith("._"):
            continue
        stem = csv_path.name[: -len(SUMMARY_SUFFIX)]
        dat_path = csv_path.with_name(f"{stem}.dat")
        rpoint_path = csv_path.with_name(f"{stem}{RPOINT_SUFFIX}")
        records.append(
            RecordPaths(
                stem=stem,
                csv_path=csv_path,
                dat_path=dat_path,
                rpoint_path=rpoint_path,
            )
        )
    return records


def parse_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if not text:
            return default
        return int(float(text))
    except Exception:
        return default


def parse_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if not text:
            return default
        return float(text)
    except Exception:
        return default


def parse_duration_hhmm(value: str) -> int | None:
    try:
        h, m = str(value).strip().split(":")[:2]
        return int(h) * 3600 + int(m) * 60
    except Exception:
        return None


def parse_start_datetime(value: str) -> datetime | None:
    text = str(value).strip()
    for fmt in ("%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return None


def read_summary_csv(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="gbk", newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        raise ValueError(f"CSV has fewer than 2 rows: {path}")
    values = rows[1]
    if len(values) < len(CSV_COLUMNS):
        values = values + [""] * (len(CSV_COLUMNS) - len(values))
    elif len(values) > len(CSV_COLUMNS):
        values = values[:15] + [",".join(values[15:])]

    row = dict(zip(CSV_COLUMNS, values, strict=True))
    row["age"] = parse_int(row.get("age"))
    row["n_channels"] = parse_int(row.get("n_channels"), N_CHANNELS)
    row["total_beats"] = parse_int(row.get("total_beats"), 0)
    row["mean_hr"] = parse_int(row.get("mean_hr"))
    row["min_hr"] = parse_int(row.get("min_hr"))
    row["max_hr"] = parse_int(row.get("max_hr"))
    row["pvc_count"] = parse_int(row.get("pvc_count"), 0)
    row["pvc_pct"] = parse_float(row.get("pvc_pct"), 0.0)
    row["svpc_count"] = parse_int(row.get("svpc_count"), 0)
    row["svpc_pct"] = parse_float(row.get("svpc_pct"), 0.0)
    row["duration_sec"] = parse_duration_hhmm(str(row.get("duration_hhmm", "")))
    row["start_dt"] = parse_start_datetime(str(row.get("start_datetime", "")))
    row["conclusion"] = str(row.get("conclusion", "")).strip()
    return row


def read_rpoint_property(path: Path) -> tuple[list[tuple[float, str]], int]:
    beats: list[tuple[float, str]] = []
    bad_lines = 0
    with path.open("r", encoding="ascii", errors="ignore") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            if ":" not in text:
                bad_lines += 1
                continue
            time_text, label = text.split(":", 1)
            try:
                beats.append((float(time_text), label.strip()))
            except ValueError:
                bad_lines += 1
    return beats, bad_lines


def count_pattern(labels: list[str], pattern: list[str]) -> int:
    """Count non-overlapping runs matching repeated pattern starts.

    For bigeminy, pattern is ["V", "N"], and an episode requires at least
    V-N-V, i.e. at least two V starts separated by the full pattern stride.
    For trigeminy, pattern is ["V", "N", "N"], requiring V-N-N-V.
    """
    episodes = 0
    i = 0
    stride = len(pattern)
    while i < len(labels):
        if labels[i : i + stride] == pattern:
            starts = 1
            j = i + stride
            while labels[j : j + stride] == pattern:
                starts += 1
                j += stride
            if j < len(labels) and labels[j] == pattern[0] and starts >= 1:
                episodes += 1
                i = j + 1
                continue
        i += 1
    return episodes


def count_repeated_v_pattern(labels: list[str], n_normals_between: int) -> int:
    """Count V-N*-V episodes, matching DMS bigeminy/trigeminy in the sample."""
    episodes = 0
    i = 0
    step = n_normals_between + 1
    while i < len(labels):
        if labels[i] == "V":
            j = i
            v_count = 1
            while (
                j + step < len(labels)
                and labels[j + step] == "V"
                and all(labels[j + d] == "N" for d in range(1, step))
            ):
                j += step
                v_count += 1
            if v_count >= 2:
                episodes += 1
                i = j + 1
                continue
        i += 1
    return episodes


def wall_time(start_dt: datetime | None, elapsed_sec: float | None) -> str:
    if start_dt is None or elapsed_sec is None:
        return ""
    return (start_dt + timedelta(seconds=float(elapsed_sec))).strftime("%H:%M:%S")


def compute_rpoint_stats(
    beats: list[tuple[float, str]],
    *,
    pvc_labels: set[str],
    svpc_labels: set[str],
    exclude_total_labels: set[str],
    duration_sec: int | None,
    start_dt: datetime | None,
) -> dict[str, Any]:
    labels = [label for _, label in beats]
    label_counts = Counter(labels)
    valid = [(t, label) for t, label in beats if label not in exclude_total_labels]
    valid_total = len(valid)
    all_total = len(beats)
    pvc_count = sum(1 for _, label in beats if label in pvc_labels)
    svpc_count = sum(1 for _, label in beats if label in svpc_labels)
    pvc_pct = 100.0 * pvc_count / valid_total if valid_total else None
    svpc_pct = 100.0 * svpc_count / valid_total if valid_total else None
    mean_hr = 60.0 * valid_total / duration_sec if duration_sec else None

    max_nn_rr = None
    max_nn_time = None
    min_plausible_nn_rr = None
    min_plausible_nn_time = None
    rr_gt_2s = 0
    rr_gt_3s = 0
    for idx in range(1, len(beats)):
        prev_t, prev_label = beats[idx - 1]
        cur_t, cur_label = beats[idx]
        rr = cur_t - prev_t
        if prev_label == "N" and cur_label == "N":
            if max_nn_rr is None or rr > max_nn_rr:
                max_nn_rr = rr
                max_nn_time = cur_t
            if 0.3 <= rr <= 2.5:
                if min_plausible_nn_rr is None or rr < min_plausible_nn_rr:
                    min_plausible_nn_rr = rr
                    min_plausible_nn_time = cur_t
            if rr > 2.0:
                rr_gt_2s += 1
            if rr > 3.0:
                rr_gt_3s += 1

    bigeminy_runs = count_repeated_v_pattern(labels, n_normals_between=1)
    trigeminy_runs = count_repeated_v_pattern(labels, n_normals_between=2)
    couplet_runs = 0
    triplet_or_more_runs = 0
    i = 0
    while i < len(labels):
        if labels[i] == "V":
            j = i
            while j < len(labels) and labels[j] == "V":
                j += 1
            run_len = j - i
            if run_len == 2:
                couplet_runs += 1
            elif run_len >= 3:
                triplet_or_more_runs += 1
            i = j
        else:
            i += 1

    return {
        "rpoint_rows": all_total,
        "rpoint_valid_total": valid_total,
        "rpoint_excluded_total": all_total - valid_total,
        "rpoint_label_counts": dict(sorted(label_counts.items())),
        "rpoint_pvc_count": pvc_count,
        "rpoint_pvc_pct": pvc_pct,
        "rpoint_svpc_count": svpc_count,
        "rpoint_svpc_pct": svpc_pct,
        "rpoint_mean_hr_by_count": mean_hr,
        "rpoint_max_nn_rr": max_nn_rr,
        "rpoint_max_nn_rr_time_sec": max_nn_time,
        "rpoint_max_nn_rr_wall_time": wall_time(start_dt, max_nn_time),
        "rpoint_max_hr_from_min_plausible_nn": (
            60.0 / min_plausible_nn_rr if min_plausible_nn_rr else None
        ),
        "rpoint_max_hr_time_sec": min_plausible_nn_time,
        "rpoint_max_hr_wall_time": wall_time(start_dt, min_plausible_nn_time),
        "rpoint_rr_gt_2s": rr_gt_2s,
        "rpoint_rr_gt_3s": rr_gt_3s,
        "rpoint_bigeminy_runs": bigeminy_runs,
        "rpoint_trigeminy_runs": trigeminy_runs,
        "rpoint_couplet_runs": couplet_runs,
        "rpoint_triplet_or_more_runs": triplet_or_more_runs,
    }


def extract_report_numbers(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not text:
        return result
    if re.search(r"\[[A-Z_]+\]", text):
        result["report_has_template_placeholders"] = 1
        return result
    result["report_has_template_placeholders"] = 0

    patterns = {
        "report_total_beats": r"分析的总心搏数为\s*(\d+)\s*个",
        "report_mean_hr": r"平均心率\s*[：:]\s*(\d+)\s*bpm",
        "report_max_hr": r"最快心率\s*[：:]\s*(\d+)\s*bpm",
        "report_min_hr": r"最慢心率\s*[：:]\s*(\d+)\s*bpm",
        "report_pvc_count": r"室性早搏\s*[：:]\s*(?:有)?\s*(\d+)\s*次",
        "report_svpc_count": r"(?:房性早搏|室上性早搏)\s*[：:]\s*(?:有)?\s*(?:单个)?\s*(\d+)\s*次",
        "report_bigeminy_runs": r"室早二联律\s*[：:]\s*(\d+)\s*阵",
        "report_trigeminy_runs": r"室早三联律\s*[：:]\s*(\d+)\s*阵",
    }
    for key, pat in patterns.items():
        match = re.search(pat, text)
        if match:
            result[key] = int(match.group(1))

    rr_match = re.search(r"最长\s*R-?R\s*间期[^\d]*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if rr_match:
        result["report_longest_rr"] = float(rr_match.group(1))

    time_patterns = {
        "report_max_hr_time": r"最快心率[^\d]*\d+\s*bpm?\s*见于\s*(\d{1,2}:\d{2}(?::\d{2})?)",
        "report_min_hr_time": r"最慢心率[^\d]*\d+\s*bpm?\s*见于\s*(\d{1,2}:\d{2}(?::\d{2})?)",
        "report_longest_rr_time": r"最长\s*R-?R\s*间期[^\d]*(?:\d+(?:\.\d+)?)\s*秒?\s*见于\s*(\d{1,2}:\d{2}(?::\d{2})?)",
    }
    for key, pat in time_patterns.items():
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            result[key] = match.group(1)

    result["report_mentions_st_t"] = int(bool(re.search(r"ST-?T|ST段", text, re.IGNORECASE)))
    result["report_negates_st_t"] = int(bool(re.search(r"(?:未见|无|未发现).{0,8}(?:ST-?T|ST段)", text, re.IGNORECASE)))
    result["report_mentions_af"] = int(bool(re.search(r"房颤|心房颤动", text)))
    result["report_mentions_pause"] = int(bool(re.search(r"停搏|长间歇|最长\s*R-?R", text, re.IGNORECASE)))
    result["report_mentions_av_block"] = int(bool(re.search(r"房室传导阻滞|AVB|传导阻滞", text, re.IGNORECASE)))
    return result


def wilson_interval(k: int, n: int, z: float = 1.96) -> dict[str, float | int]:
    if n <= 0:
        return {"count": k, "n": n, "rate": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return {
        "count": k,
        "n": n,
        "rate": p,
        "ci_low": max(0.0, center - half),
        "ci_high": min(1.0, center + half),
    }


def add_mismatch(
    mismatches: list[dict[str, Any]],
    *,
    stem: str,
    pid: str,
    field: str,
    expected_from: str,
    expected: Any,
    observed_from: str,
    observed: Any,
    tolerance: float,
) -> None:
    try:
        diff = abs(float(expected) - float(observed))
    except Exception:
        diff = None
    mismatches.append(
        {
            "stem": stem,
            "pid": pid,
            "field": field,
            "expected_from": expected_from,
            "expected": expected,
            "observed_from": observed_from,
            "observed": observed,
            "abs_diff": diff,
            "tolerance": tolerance,
        }
    )


def compare_numeric(
    mismatches: list[dict[str, Any]],
    stats: dict[str, int],
    *,
    stem: str,
    pid: str,
    field: str,
    expected_from: str,
    expected: Any,
    observed_from: str,
    observed: Any,
    tolerance: float,
) -> bool:
    if expected is None or observed is None:
        return False
    stats[f"{field}__denom"] += 1
    try:
        ok = abs(float(expected) - float(observed)) <= tolerance
    except Exception:
        ok = str(expected) == str(observed)
    if not ok:
        stats[f"{field}__mismatch"] += 1
        add_mismatch(
            mismatches,
            stem=stem,
            pid=pid,
            field=field,
            expected_from=expected_from,
            expected=expected,
            observed_from=observed_from,
            observed=observed,
            tolerance=tolerance,
        )
    return not ok


def dat_size_stats(path: Path, duration_sec: int | None, n_channels: int | None) -> dict[str, Any]:
    if not path.exists():
        return {"has_dat": 0, "dat_size_bytes": None, "expected_dat_size_bytes": None, "dat_size_match": None}
    size = path.stat().st_size
    expected = None
    match = None
    if duration_sec is not None and n_channels:
        expected = int(duration_sec * SAMPLE_RATE * n_channels)
        match = int(size == expected)
    return {
        "has_dat": 1,
        "dat_size_bytes": size,
        "expected_dat_size_bytes": expected,
        "dat_size_match": match,
    }


def flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in row.items():
        if isinstance(value, (dict, list, tuple)):
            out[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        elif isinstance(value, float):
            out[key] = round(value, 6)
        else:
            out[key] = value
    return out


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pvc_labels = split_label_set(args.pvc_labels)
    svpc_labels = split_label_set(args.svpc_labels)
    exclude_total_labels = split_label_set(args.exclude_total_labels)

    records = scan_records(data_dir, args.recursive)
    if args.limit is not None:
        records = records[: args.limit]

    audit_rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    mismatch_stats: dict[str, int] = defaultdict(int)
    global_label_counts: Counter[str] = Counter()
    record_label_presence: Counter[str] = Counter()
    parse_errors = []
    report_template_placeholders = 0

    for rec in records:
        row: dict[str, Any] = {
            "stem": rec.stem,
            "csv_path": str(rec.csv_path),
            "dat_path": str(rec.dat_path),
            "rpoint_path": str(rec.rpoint_path),
        }
        try:
            summary = read_summary_csv(rec.csv_path)
            pid = str(summary.get("pid") or rec.stem)
            row.update(
                {
                    "pid": pid,
                    "name": summary.get("name"),
                    "sex": summary.get("sex"),
                    "age": summary.get("age"),
                    "duration_hhmm": summary.get("duration_hhmm"),
                    "duration_sec": summary.get("duration_sec"),
                    "start_datetime": summary.get("start_datetime"),
                    "csv_total_beats": summary.get("total_beats"),
                    "csv_mean_hr": summary.get("mean_hr"),
                    "csv_min_hr": summary.get("min_hr"),
                    "csv_max_hr": summary.get("max_hr"),
                    "csv_pvc_count": summary.get("pvc_count"),
                    "csv_pvc_pct": summary.get("pvc_pct"),
                    "csv_svpc_count": summary.get("svpc_count"),
                    "csv_svpc_pct": summary.get("svpc_pct"),
                    "conclusion": summary.get("conclusion"),
                }
            )
            row.update(dat_size_stats(rec.dat_path, summary.get("duration_sec"), summary.get("n_channels")))

            report_numbers = extract_report_numbers(summary.get("conclusion", ""))
            row.update(report_numbers)
            if report_numbers.get("report_has_template_placeholders") == 1:
                report_template_placeholders += 1

            if rec.rpoint_path.exists():
                beats, bad_lines = read_rpoint_property(rec.rpoint_path)
                row["has_rpoint"] = 1
                row["rpoint_bad_lines"] = bad_lines
                rstats = compute_rpoint_stats(
                    beats,
                    pvc_labels=pvc_labels,
                    svpc_labels=svpc_labels,
                    exclude_total_labels=exclude_total_labels,
                    duration_sec=summary.get("duration_sec"),
                    start_dt=summary.get("start_dt"),
                )
                row.update(rstats)
                global_label_counts.update(rstats["rpoint_label_counts"])
                for label, count in rstats["rpoint_label_counts"].items():
                    if count:
                        record_label_presence[label] += 1

                compare_numeric(
                    mismatches,
                    mismatch_stats,
                    stem=rec.stem,
                    pid=pid,
                    field="csv_total_vs_rpoint_valid_total",
                    expected_from="RPointProperty",
                    expected=rstats["rpoint_valid_total"],
                    observed_from="HolterSummary.csv",
                    observed=summary.get("total_beats"),
                    tolerance=0,
                )
                compare_numeric(
                    mismatches,
                    mismatch_stats,
                    stem=rec.stem,
                    pid=pid,
                    field="csv_pvc_count_vs_rpoint",
                    expected_from="RPointProperty",
                    expected=rstats["rpoint_pvc_count"],
                    observed_from="HolterSummary.csv",
                    observed=summary.get("pvc_count"),
                    tolerance=0,
                )
                compare_numeric(
                    mismatches,
                    mismatch_stats,
                    stem=rec.stem,
                    pid=pid,
                    field="csv_svpc_count_vs_rpoint",
                    expected_from="RPointProperty",
                    expected=rstats["rpoint_svpc_count"],
                    observed_from="HolterSummary.csv",
                    observed=summary.get("svpc_count"),
                    tolerance=0,
                )
                compare_numeric(
                    mismatches,
                    mismatch_stats,
                    stem=rec.stem,
                    pid=pid,
                    field="csv_pvc_pct_vs_rpoint",
                    expected_from="RPointProperty",
                    expected=rstats["rpoint_pvc_pct"],
                    observed_from="HolterSummary.csv",
                    observed=summary.get("pvc_pct"),
                    tolerance=args.pct_tolerance,
                )
                compare_numeric(
                    mismatches,
                    mismatch_stats,
                    stem=rec.stem,
                    pid=pid,
                    field="csv_svpc_pct_vs_rpoint",
                    expected_from="RPointProperty",
                    expected=rstats["rpoint_svpc_pct"],
                    observed_from="HolterSummary.csv",
                    observed=summary.get("svpc_pct"),
                    tolerance=args.pct_tolerance,
                )
                compare_numeric(
                    mismatches,
                    mismatch_stats,
                    stem=rec.stem,
                    pid=pid,
                    field="csv_mean_hr_vs_rpoint_count",
                    expected_from="RPointProperty",
                    expected=rstats["rpoint_mean_hr_by_count"],
                    observed_from="HolterSummary.csv",
                    observed=summary.get("mean_hr"),
                    tolerance=args.hr_tolerance,
                )

                if "report_longest_rr" in report_numbers:
                    compare_numeric(
                        mismatches,
                        mismatch_stats,
                        stem=rec.stem,
                        pid=pid,
                        field="report_longest_rr_vs_rpoint_nn",
                        expected_from="RPointProperty",
                        expected=rstats["rpoint_max_nn_rr"],
                        observed_from="Conclusion text",
                        observed=report_numbers.get("report_longest_rr"),
                        tolerance=args.rr_tolerance,
                    )
                if "report_bigeminy_runs" in report_numbers:
                    compare_numeric(
                        mismatches,
                        mismatch_stats,
                        stem=rec.stem,
                        pid=pid,
                        field="report_bigeminy_vs_rpoint",
                        expected_from="RPointProperty",
                        expected=rstats["rpoint_bigeminy_runs"],
                        observed_from="Conclusion text",
                        observed=report_numbers.get("report_bigeminy_runs"),
                        tolerance=0,
                    )
                if "report_trigeminy_runs" in report_numbers:
                    compare_numeric(
                        mismatches,
                        mismatch_stats,
                        stem=rec.stem,
                        pid=pid,
                        field="report_trigeminy_vs_rpoint",
                        expected_from="RPointProperty",
                        expected=rstats["rpoint_trigeminy_runs"],
                        observed_from="Conclusion text",
                        observed=report_numbers.get("report_trigeminy_runs"),
                        tolerance=0,
                    )
            else:
                row["has_rpoint"] = 0
                row["rpoint_bad_lines"] = None

            for report_key, csv_key, tol in [
                ("report_total_beats", "total_beats", 0),
                ("report_mean_hr", "mean_hr", args.hr_tolerance),
                ("report_max_hr", "max_hr", args.hr_tolerance),
                ("report_min_hr", "min_hr", args.hr_tolerance),
                ("report_pvc_count", "pvc_count", 0),
                ("report_svpc_count", "svpc_count", 0),
            ]:
                if report_key in report_numbers:
                    compare_numeric(
                        mismatches,
                        mismatch_stats,
                        stem=rec.stem,
                        pid=pid,
                        field=f"{report_key}_vs_csv",
                        expected_from="HolterSummary.csv",
                        expected=summary.get(csv_key),
                        observed_from="Conclusion text",
                        observed=report_numbers.get(report_key),
                        tolerance=tol,
                    )

        except Exception as exc:
            row["parse_error"] = str(exc)
            parse_errors.append({"stem": rec.stem, "error": str(exc)})

        audit_rows.append(row)

    audit_csv_path = output_dir / "audit_records.csv"
    mismatch_csv_path = output_dir / "audit_mismatches.csv"
    summary_json_path = output_dir / "audit_summary.json"

    all_fieldnames = sorted({key for row in audit_rows for key in row.keys()})
    with audit_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fieldnames)
        writer.writeheader()
        for row in audit_rows:
            writer.writerow(flatten_for_csv(row))

    mismatch_fieldnames = [
        "stem",
        "pid",
        "field",
        "expected_from",
        "expected",
        "observed_from",
        "observed",
        "abs_diff",
        "tolerance",
    ]
    with mismatch_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=mismatch_fieldnames)
        writer.writeheader()
        for item in mismatches:
            writer.writerow(flatten_for_csv(item))

    n_records = len(audit_rows)
    has_rpoint = sum(1 for row in audit_rows if row.get("has_rpoint") == 1)
    has_dat = sum(1 for row in audit_rows if row.get("has_dat") == 1)
    dat_size_match = sum(1 for row in audit_rows if row.get("dat_size_match") == 1)
    dat_size_denom = sum(1 for row in audit_rows if row.get("dat_size_match") is not None)

    mismatch_summary = {}
    fields = sorted({key.rsplit("__", 1)[0] for key in mismatch_stats if key.endswith("__denom")})
    for field in fields:
        k = mismatch_stats.get(f"{field}__mismatch", 0)
        n = mismatch_stats.get(f"{field}__denom", 0)
        mismatch_summary[field] = wilson_interval(k, n)

    record_label_summary = {
        label: wilson_interval(count, has_rpoint)
        for label, count in sorted(record_label_presence.items())
    }

    json_summary = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "n_records": n_records,
        "parse_errors": parse_errors,
        "file_completeness": {
            "rpoint_available": wilson_interval(has_rpoint, n_records),
            "dat_available": wilson_interval(has_dat, n_records),
            "dat_size_match": wilson_interval(dat_size_match, dat_size_denom),
        },
        "beat_label_counts_global": dict(sorted(global_label_counts.items())),
        "beat_label_record_prevalence": record_label_summary,
        "mismatch_summary": mismatch_summary,
        "mismatch_total_rows": len(mismatches),
        "report_template_placeholders": wilson_interval(report_template_placeholders, n_records),
        "config": {
            "sample_rate": SAMPLE_RATE,
            "n_channels_default": N_CHANNELS,
            "pvc_labels": sorted(pvc_labels),
            "svpc_labels": sorted(svpc_labels),
            "exclude_total_labels": sorted(exclude_total_labels),
            "pct_tolerance": args.pct_tolerance,
            "hr_tolerance": args.hr_tolerance,
            "rr_tolerance": args.rr_tolerance,
        },
    }
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2, ensure_ascii=False)

    print(f"Scanned records: {n_records}")
    print(f"RPointProperty available: {has_rpoint}/{n_records}")
    print(f"DAT available: {has_dat}/{n_records}")
    print(f"Mismatch rows: {len(mismatches)}")
    print(f"Audit records: {audit_csv_path}")
    print(f"Mismatches:    {mismatch_csv_path}")
    print(f"Summary JSON:  {summary_json_path}")


if __name__ == "__main__":
    main()
