"""PyTorch dataset for beat-level tokenizer pretraining and downstream tasks."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .holter_record import HolterRecord, SAMPLE_RATE, BEAT_TYPES


class BeatTokenizerDataset(Dataset):
    """Yields individual beat windows + metadata for beat-level objectives."""

    def __init__(
        self,
        records: list[HolterRecord],
        pre_samples: int = 24,
        post_samples: int = 56,
        augment: bool = False,
    ):
        self.pre = pre_samples
        self.post = post_samples
        self.window_len = pre_samples + post_samples
        self.augment = augment

        self.windows: list[np.ndarray] = []
        self.metas: list[np.ndarray] = []
        self.labels: list[int] = []
        self.record_ids: list[str] = []

        for rec in records:
            beats = rec.load_beats()
            wins, meta, valid = rec.extract_beat_windows(pre_samples, post_samples)
            for i in range(len(beats.times)):
                if not valid[i]:
                    continue
                self.windows.append(wins[i])
                self.metas.append(meta[i])
                self.labels.append(int(beats.labels[i]))
                self.record_ids.append(rec.record_id)
            rec.free()

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        win = self.windows[idx].copy()  # (window_len, 3)
        meta = self.metas[idx].copy()   # (6,)
        label = self.labels[idx]

        if self.augment:
            win = self._augment(win)

        return {
            "waveform": torch.from_numpy(win).float(),       # (80, 3)
            "metadata": torch.from_numpy(meta).float(),       # (6,)
            "label": torch.tensor(label, dtype=torch.long),   # scalar
        }

    def _augment(self, win: np.ndarray) -> np.ndarray:
        # amplitude scale
        scale = np.random.uniform(0.9, 1.1)
        win = win * scale
        # gaussian noise
        sigma = np.random.uniform(0, 0.02)
        win = win + np.random.randn(*win.shape).astype(np.float32) * sigma
        # baseline wander (low-freq sinusoid)
        amp = np.random.uniform(0, 0.05)
        freq = np.random.uniform(0.1, 0.5)
        t = np.linspace(0, 1, win.shape[0], dtype=np.float32)
        wander = amp * np.sin(2 * np.pi * freq * t)
        win = win + wander[:, None]
        # lead dropout
        if np.random.random() < 0.1:
            ch = np.random.randint(0, 3)
            win[:, ch] = 0.0
        # time jitter (shift ±4 samples)
        shift = np.random.randint(-4, 5)
        if shift != 0:
            win = np.roll(win, shift, axis=0)
            if shift > 0:
                win[:shift] = 0.0
            else:
                win[shift:] = 0.0
        return np.clip(win, -5.0, 5.0)
