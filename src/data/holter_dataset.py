"""Full-day Holter dataset for multi-scale pretraining."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .holter_record import HolterRecord, SAMPLE_RATE, BEAT_TYPES


class HolterPretrainDataset(Dataset):
    """One item = one full 24h recording, yielding beat windows + episodes + rhythm tokens."""

    def __init__(
        self,
        records: list[HolterRecord],
        episode_len: int = 64,
        pre_samples: int = 24,
        post_samples: int = 56,
        max_beats: int = 150_000,
        augment: bool = False,
        day_stats_from_report: bool = True,
    ):
        self.records = records
        self.episode_len = episode_len
        self.pre = pre_samples
        self.post = post_samples
        self.window_len = pre_samples + post_samples
        self.max_beats = max_beats
        self.augment = augment
        self.day_stats_from_report = day_stats_from_report

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        beats = rec.load_beats()
        windows, metadata, valid = rec.extract_beat_windows(self.pre, self.post)

        n_beats = min(len(beats.times), self.max_beats)
        windows = windows[:n_beats]
        metadata = metadata[:n_beats]
        valid = valid[:n_beats]
        beat_labels = beats.labels[:n_beats].copy()
        beat_times = beats.times[:n_beats].copy()

        # --- rhythm tokens ---
        rr = np.diff(beat_times)
        rr = np.concatenate([rr, [rr[-1] if len(rr) > 0 else 0.8]])
        rr_prev = np.concatenate([[rr[0]], rr[:-1]])
        rr_bins = self._quantize_rr(rr, n_bins=32)
        rr_prev_bins = self._quantize_rr(rr_prev, n_bins=32)
        hour_of_day = (beat_times % 86400) / 3600.0
        hour_sin = np.sin(2 * np.pi * hour_of_day / 24.0).astype(np.float32)
        hour_cos = np.cos(2 * np.pi * hour_of_day / 24.0).astype(np.float32)

        # --- episode boundaries ---
        n_episodes = n_beats // self.episode_len
        ep_n_beats = n_episodes * self.episode_len

        # --- day statistics (auto-derived from annotations) ---
        day_stats = self._compute_day_stats(beat_times, beat_labels, rr)

        # --- report concept labels ---
        concept_labels = None
        if self.day_stats_from_report:
            try:
                from .report_concepts import ReportConceptExtractor
                report = rec.load_report()
                extractor = ReportConceptExtractor()
                concept_labels = np.array(extractor.extract_vector(report.conclusion), dtype=np.float32)
            except Exception:
                concept_labels = None

        if self.augment:
            windows = self._augment_batch(windows)

        result = {
            "record_id": rec.record_id,
            "n_beats": n_beats,
            "n_episodes": n_episodes,
            # beat-level
            "windows": torch.from_numpy(windows).float(),           # (n_beats, 80, 3)
            "metadata": torch.from_numpy(metadata).float(),         # (n_beats, 6)
            "beat_labels": torch.from_numpy(beat_labels).long(),    # (n_beats,)
            "valid_mask": torch.from_numpy(valid).bool(),           # (n_beats,)
            # rhythm tokens
            "rr_bins": torch.from_numpy(rr_bins).long(),            # (n_beats,)
            "rr_prev_bins": torch.from_numpy(rr_prev_bins).long(), # (n_beats,)
            "hour_sin": torch.from_numpy(hour_sin).float(),         # (n_beats,)
            "hour_cos": torch.from_numpy(hour_cos).float(),         # (n_beats,)
            # day-level
            "day_stats": torch.from_numpy(day_stats).float(),       # (12,)
        }
        if concept_labels is not None:
            result["concept_labels"] = torch.from_numpy(concept_labels).float()

        rec.free()
        return result

    @staticmethod
    def _quantize_rr(rr: np.ndarray, n_bins: int = 32) -> np.ndarray:
        log_rr = np.log(np.clip(rr, 0.15, 3.0))
        log_min, log_max = np.log(0.15), np.log(3.0)
        bins = ((log_rr - log_min) / (log_max - log_min) * (n_bins - 1)).astype(np.int32)
        return np.clip(bins, 0, n_bins - 1)

    @staticmethod
    def _compute_day_stats(times: np.ndarray, labels: np.ndarray, rr: np.ndarray) -> np.ndarray:
        n = len(times)
        rr_real = rr[:-1]  # exclude the duplicated last element
        rr_nn = rr_real[(labels[:-1] == 0) & (labels[1:] == 0)] if n > 1 else rr_real
        if len(rr_nn) == 0:
            rr_nn = rr

        mean_hr = 60.0 / max(np.mean(rr), 0.2)
        min_hr = 60.0 / max(np.max(rr), 0.2)
        max_hr = 60.0 / max(np.min(rr), 0.2)
        sdnn = np.std(rr_nn) * 1000 if len(rr_nn) > 1 else 0.0
        rmssd = np.sqrt(np.mean(np.diff(rr_nn) ** 2)) * 1000 if len(rr_nn) > 2 else 0.0
        total_beats = float(n)
        pvc_count = float(np.sum(labels == 1))
        pvc_burden = pvc_count / max(n, 1) * 100.0
        longest_rr = float(np.max(rr)) if len(rr) > 0 else 0.0

        # bigeminy: NVNV pattern ≥ 6 beats
        bigeminy_count = 0.0
        label_str = "".join(["N" if l == 0 else "V" if l == 1 else "F" for l in labels])
        import re
        bigeminy_count = float(len(re.findall(r"(?=NVNVNV)", label_str)))

        # trigeminy: NNVNNV pattern ≥ 6 beats
        trigeminy_count = float(len(re.findall(r"(?=NNVNNV)", label_str)))

        # ventricular runs: ≥3 consecutive V
        v_run_count = float(len(re.findall(r"V{3,}", label_str)))

        return np.array([
            mean_hr, min_hr, max_hr, sdnn, rmssd,
            total_beats, pvc_count, pvc_burden, longest_rr,
            bigeminy_count, trigeminy_count, v_run_count,
        ], dtype=np.float32)

    def _augment_batch(self, windows: np.ndarray) -> np.ndarray:
        scale = np.random.uniform(0.9, 1.1)
        windows = windows * scale
        sigma = np.random.uniform(0, 0.02)
        if sigma > 0:
            windows = windows + np.random.randn(*windows.shape).astype(np.float32) * sigma
        amp = np.random.uniform(0, 0.05)
        if amp > 0:
            freq = np.random.uniform(0.1, 0.5)
            t = np.linspace(0, 1, windows.shape[1], dtype=np.float32)
            wander = amp * np.sin(2 * np.pi * freq * t)
            windows = windows + wander[None, :, None]
        if np.random.random() < 0.1:
            ch = np.random.randint(0, 3)
            windows[:, :, ch] = 0.0
        return np.clip(windows, -5.0, 5.0)


def collate_holter(batch: list[dict]) -> dict:
    """Custom collate: pad beat sequences to max length in batch."""
    max_beats = max(b["n_beats"] for b in batch)
    bs = len(batch)
    window_len = batch[0]["windows"].shape[1]

    out = {
        "record_id": [b["record_id"] for b in batch],
        "n_beats": torch.tensor([b["n_beats"] for b in batch]),
        "n_episodes": torch.tensor([b["n_episodes"] for b in batch]),
        "windows": torch.zeros(bs, max_beats, window_len, 3),
        "metadata": torch.zeros(bs, max_beats, 6),
        "beat_labels": torch.full((bs, max_beats), -1, dtype=torch.long),
        "valid_mask": torch.zeros(bs, max_beats, dtype=torch.bool),
        "rr_bins": torch.zeros(bs, max_beats, dtype=torch.long),
        "rr_prev_bins": torch.zeros(bs, max_beats, dtype=torch.long),
        "hour_sin": torch.zeros(bs, max_beats),
        "hour_cos": torch.zeros(bs, max_beats),
        "day_stats": torch.stack([b["day_stats"] for b in batch]),
        "padding_mask": torch.ones(bs, max_beats, dtype=torch.bool),
    }

    has_concepts = all("concept_labels" in b for b in batch)
    if has_concepts:
        out["concept_labels"] = torch.stack([b["concept_labels"] for b in batch])

    for i, b in enumerate(batch):
        n = b["n_beats"]
        out["windows"][i, :n] = b["windows"]
        out["metadata"][i, :n] = b["metadata"]
        out["beat_labels"][i, :n] = b["beat_labels"]
        out["valid_mask"][i, :n] = b["valid_mask"]
        out["rr_bins"][i, :n] = b["rr_bins"]
        out["rr_prev_bins"][i, :n] = b["rr_prev_bins"]
        out["hour_sin"][i, :n] = b["hour_sin"]
        out["hour_cos"][i, :n] = b["hour_cos"]
        out["padding_mask"][i, :n] = False

    return out
