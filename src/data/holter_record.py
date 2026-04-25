"""Single Holter recording: waveform + beats + report."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


SAMPLE_RATE = 128
N_CHANNELS = 3
BEAT_TYPES = {"N": 0, "V": 1, "F": 2}


@dataclass
class BeatAnnotation:
    times: np.ndarray       # (n_beats,) seconds from recording start
    labels: np.ndarray      # (n_beats,) int: 0=N, 1=V, 2=F
    raw_labels: list[str]   # original char labels


@dataclass
class ReportSummary:
    name: str
    pid: str
    sex: str
    age: int
    n_channels: int
    start_datetime: str
    duration_hhmm: str
    total_beats: int
    mean_hr: int
    min_hr: int
    max_hr: int
    pvc_count: int
    pvc_pct: float
    svpc_count: int
    svpc_pct: float
    conclusion: str


@dataclass
class HolterRecord:
    record_id: str
    dat_path: Path
    summary_path: Path
    rpoint_path: Path

    _ecg: Optional[np.ndarray] = field(default=None, repr=False)
    _beats: Optional[BeatAnnotation] = field(default=None, repr=False)
    _report: Optional[ReportSummary] = field(default=None, repr=False)

    @classmethod
    def from_dat_path(cls, dat_path: str | Path) -> "HolterRecord":
        dat_path = Path(dat_path)
        stem = dat_path.stem  # e.g. 202108120755_竺春芳_D07984510
        parent = dat_path.parent
        return cls(
            record_id=stem,
            dat_path=dat_path,
            summary_path=parent / f"{stem}_HolterSummary.csv",
            rpoint_path=parent / f"{stem}_RPointProperty.txt",
        )

    @classmethod
    def discover(cls, data_dir: str | Path) -> list["HolterRecord"]:
        data_dir = Path(data_dir)
        records = []
        for dat in sorted(data_dir.glob("*.dat")):
            if dat.name.startswith("._"):
                continue
            rec = cls.from_dat_path(dat)
            if rec.rpoint_path.exists():
                records.append(rec)
        return records

    # --- ECG waveform ---

    def load_ecg_raw(self) -> np.ndarray:
        raw = np.fromfile(self.dat_path, dtype=np.uint8)
        n_samples = len(raw) // N_CHANNELS
        return raw[: n_samples * N_CHANNELS].reshape(-1, N_CHANNELS)

    def load_ecg(self, normalize: bool = True) -> np.ndarray:
        if self._ecg is not None:
            return self._ecg
        ecg = self.load_ecg_raw().astype(np.float32)
        if normalize:
            median = np.median(ecg, axis=0, keepdims=True)
            mad = np.median(np.abs(ecg - median), axis=0, keepdims=True)
            mad = np.maximum(mad * 1.4826, 1e-6)
            ecg = np.clip((ecg - median) / mad, -5.0, 5.0)
        self._ecg = ecg
        return ecg

    @property
    def n_samples(self) -> int:
        return self.dat_path.stat().st_size // N_CHANNELS

    @property
    def duration_seconds(self) -> float:
        return self.n_samples / SAMPLE_RATE

    # --- Beat annotations ---

    def load_beats(self) -> BeatAnnotation:
        if self._beats is not None:
            return self._beats
        times, labels_int, labels_raw = [], [], []
        with open(self.rpoint_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                t_str, lbl = line.split(":")
                times.append(float(t_str))
                labels_raw.append(lbl)
                labels_int.append(BEAT_TYPES.get(lbl, -1))
        self._beats = BeatAnnotation(
            times=np.array(times, dtype=np.float64),
            labels=np.array(labels_int, dtype=np.int8),
            raw_labels=labels_raw,
        )
        return self._beats

    # --- Report summary ---

    def load_report(self) -> ReportSummary:
        if self._report is not None:
            return self._report
        if not self.summary_path.exists():
            raise FileNotFoundError(f"No summary CSV: {self.summary_path}")
        with open(self.summary_path, "r", encoding="gbk") as f:
            raw = f.read()
        lines = raw.strip().split("\n")
        header = lines[0].strip().split(",")
        # conclusion field may contain commas — take everything after field 14
        values = lines[1].strip().split(",")
        n_fixed = 15
        if len(values) > n_fixed:
            values = values[:n_fixed] + [",".join(values[n_fixed:])]
        self._report = ReportSummary(
            name=values[0],
            pid=values[1],
            sex=values[2],
            age=int(values[3]),
            n_channels=int(values[4]),
            start_datetime=values[5],
            duration_hhmm=values[6],
            total_beats=int(values[7]),
            mean_hr=int(values[8]),
            min_hr=int(values[9]),
            max_hr=int(values[10]),
            pvc_count=int(values[11]),
            pvc_pct=float(values[12]),
            svpc_count=int(values[13]),
            svpc_pct=float(values[14]),
            conclusion=values[15] if len(values) > 15 else "",
        )
        return self._report

    # --- Beat windows ---

    def extract_beat_windows(
        self,
        pre_samples: int = 24,
        post_samples: int = 56,
        adaptive_clip: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract beat-centered windows from normalized ECG.

        Returns:
            windows: (n_beats, window_len, 3) float32
            metadata: (n_beats, 6) float32 — [log_rr_prev, log_rr_next, inst_hr, hour_sin, hour_cos, norm_idx]
            valid_mask: (n_beats,) bool
        """
        ecg = self.load_ecg()
        beats = self.load_beats()
        n_beats = len(beats.times)
        window_len = pre_samples + post_samples

        windows = np.zeros((n_beats, window_len, N_CHANNELS), dtype=np.float32)
        metadata = np.zeros((n_beats, 6), dtype=np.float32)
        valid = np.ones(n_beats, dtype=bool)

        rr = np.diff(beats.times)
        rr_prev = np.concatenate([[rr[0]], rr])
        rr_next = np.concatenate([rr, [rr[-1]]])

        for i in range(n_beats):
            center = int(beats.times[i] * SAMPLE_RATE)
            start = center - pre_samples
            end = center + post_samples

            if adaptive_clip:
                if i > 0:
                    prev_center = int(beats.times[i - 1] * SAMPLE_RATE)
                    max_pre = center - prev_center - post_samples
                    if max_pre < pre_samples:
                        start = max(center - max(max_pre, 4), 0)
                if i < n_beats - 1:
                    next_center = int(beats.times[i + 1] * SAMPLE_RATE)
                    max_post = next_center - center - pre_samples
                    if max_post < post_samples:
                        end = min(center + max(max_post, 4), len(ecg))

            if start < 0 or end > len(ecg):
                valid[i] = False
                start = max(start, 0)
                end = min(end, len(ecg))

            seg = ecg[start:end]
            actual_pre = center - start
            actual_post = end - center
            w_start = pre_samples - actual_pre
            w_end = w_start + len(seg)
            if w_end <= window_len and w_start >= 0:
                windows[i, w_start:w_end] = seg

            # metadata
            metadata[i, 0] = np.log(max(rr_prev[i], 0.1))
            metadata[i, 1] = np.log(max(rr_next[i], 0.1))
            metadata[i, 2] = 60.0 / max(rr_prev[i], 0.2)  # inst HR
            hour = (beats.times[i] % 86400) / 3600.0
            metadata[i, 3] = np.sin(2 * np.pi * hour / 24.0)
            metadata[i, 4] = np.cos(2 * np.pi * hour / 24.0)
            metadata[i, 5] = i / max(n_beats - 1, 1)

        return windows, metadata, valid

    def free(self):
        self._ecg = None
        self._beats = None
        self._report = None
