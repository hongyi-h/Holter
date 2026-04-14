"""PyTorch datasets for Holter SSL."""
import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset


class HolterPretrainDataset(Dataset):
    """Dataset for SSL pre-training. Loads processed .npz patient files."""

    def __init__(self, processed_dir, mode="ordered", max_windows=288, max_beats=300):
        self.processed_dir = processed_dir
        self.mode = mode
        self.max_windows = max_windows
        self.max_beats = max_beats
        self.files = sorted(glob.glob(os.path.join(processed_dir, "*.npz")))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npz files in {processed_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)
        beat_tensors = data["beat_tensors"].astype(np.float32)   # (W, N, S, C)
        masks = data["masks"].astype(bool)                       # (W, N)
        time_encodings = data["time_encodings"].astype(np.float32)  # (W, 2)
        quality_mask = data["quality_mask"].astype(bool)          # (W,)
        patient_id = str(data.get("patient_id", f"patient_{idx}"))

        W = beat_tensors.shape[0]
        if W > self.max_windows:
            beat_tensors = beat_tensors[:self.max_windows]
            masks = masks[:self.max_windows]
            time_encodings = time_encodings[:self.max_windows]
            quality_mask = quality_mask[:self.max_windows]
            W = self.max_windows

        # Apply mode
        if self.mode == "shuffled":
            perm = np.random.permutation(W)
            beat_tensors = beat_tensors[perm]
            masks = masks[perm]
            time_encodings = np.zeros_like(time_encodings)  # zero out to prevent temporal shortcut
            quality_mask = quality_mask[perm]
        elif self.mode == "snippet":
            if W > 12:
                start = np.random.randint(0, W - 12)
                beat_tensors = beat_tensors[start:start+12]
                masks = masks[start:start+12]
                time_encodings = time_encodings[start:start+12]
                quality_mask = quality_mask[start:start+12]
                W = 12

        # Window mask (all true up to W)
        window_mask = np.ones(W, dtype=bool)

        return {
            "beat_tensors": torch.from_numpy(beat_tensors),
            "beat_masks": torch.from_numpy(masks),
            "time_encodings": torch.from_numpy(time_encodings),
            "quality_mask": torch.from_numpy(quality_mask),
            "window_mask": torch.from_numpy(window_mask),
            "n_windows": W,
            "patient_id": patient_id,
        }


def collate_pretrain(batch):
    """Custom collate to pad variable-length window sequences."""
    max_w = max(b["n_windows"] for b in batch)
    B = len(batch)
    sample = batch[0]
    N = sample["beat_tensors"].shape[1]  # max_beats
    S = sample["beat_tensors"].shape[2]  # beat_samples
    C = sample["beat_tensors"].shape[3]  # channels

    bt = torch.zeros(B, max_w, N, S, C)
    bm = torch.zeros(B, max_w, N, dtype=torch.bool)
    te = torch.zeros(B, max_w, 2)
    qm = torch.zeros(B, max_w, dtype=torch.bool)
    wm = torch.zeros(B, max_w, dtype=torch.bool)

    for i, b in enumerate(batch):
        w = b["n_windows"]
        bt[i, :w] = b["beat_tensors"]
        bm[i, :w] = b["beat_masks"]
        te[i, :w] = b["time_encodings"]
        qm[i, :w] = b["quality_mask"]
        wm[i, :w] = b["window_mask"]

    return {
        "beat_tensors": bt,
        "beat_masks": bm,
        "time_encodings": te,
        "quality_mask": qm,
        "window_mask": wm,
        "n_windows": [b["n_windows"] for b in batch],
        "patient_id": [b["patient_id"] for b in batch],
    }


class HolterProbeDataset(Dataset):
    """Dataset for linear probe evaluation."""

    def __init__(self, embeddings_path, labels_path):
        emb_data = np.load(embeddings_path)
        lab_data = np.load(labels_path, allow_pickle=True)
        self.embeddings = emb_data["embeddings"].astype(np.float32)
        self.labels = lab_data["labels"]
        self.patient_ids = lab_data.get("patient_ids", np.arange(len(self.labels)))

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {
            "embedding": torch.from_numpy(self.embeddings[idx]),
            "label": self.labels[idx],
            "patient_id": str(self.patient_ids[idx]),
        }
