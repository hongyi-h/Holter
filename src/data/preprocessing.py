"""Preprocessing: windowing, quality masks, time encoding."""
import numpy as np


def circular_time_encoding(time_of_day_sec, period=86400.0):
    phase = 2.0 * np.pi * time_of_day_sec / period
    return np.array([np.sin(phase), np.cos(phase)], dtype=np.float32)


def normalize_beats(beats, method="zscore"):
    if method == "zscore":
        mean = beats.mean(axis=(1, 2), keepdims=True)
        std = beats.std(axis=(1, 2), keepdims=True) + 1e-8
        return (beats - mean) / std
    elif method == "minmax":
        bmin = beats.min(axis=(1, 2), keepdims=True)
        bmax = beats.max(axis=(1, 2), keepdims=True)
        return (beats - bmin) / (bmax - bmin + 1e-8)
    return beats


def pad_or_truncate_beats(beats, max_beats=300):
    n = beats.shape[0]
    beat_samples = beats.shape[1]
    n_channels = beats.shape[2]
    if n >= max_beats:
        return beats[:max_beats], np.ones(max_beats, dtype=bool)
    padded = np.zeros((max_beats, beat_samples, n_channels), dtype=np.float32)
    padded[:n] = beats
    mask = np.zeros(max_beats, dtype=bool)
    mask[:n] = True
    return padded, mask


def build_quality_mask(windows, quality_threshold=0.5):
    return np.array([w["mean_quality"] >= quality_threshold for w in windows], dtype=bool)


def prepare_patient_windows(windows, max_beats_per_window=300, start_time_sec=0.0):
    if not windows:
        return {"beat_tensors": [], "masks": [], "time_encodings": [], "quality_mask": np.array([])}
    beat_tensors = []
    masks = []
    time_encodings = []
    for w in windows:
        beats = normalize_beats(w["beats"])
        padded, mask = pad_or_truncate_beats(beats, max_beats_per_window)
        tod = (w["time_of_day_sec"] + start_time_sec) % 86400.0
        time_enc = circular_time_encoding(tod)
        beat_tensors.append(padded)
        masks.append(mask)
        time_encodings.append(time_enc)
    quality_mask = build_quality_mask(windows)
    return {
        "beat_tensors": np.stack(beat_tensors),
        "masks": np.stack(masks),
        "time_encodings": np.stack(time_encodings),
        "quality_mask": quality_mask,
    }
