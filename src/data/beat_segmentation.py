"""R-peak detection and beat segmentation for 3-channel Holter ECG."""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def bandpass_filter(signal, low=0.5, high=40.0, fs=128, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal, axis=0)


def detect_r_peaks(ecg_channel, fs=128):
    filtered = bandpass_filter(ecg_channel.copy(), low=5.0, high=30.0, fs=fs)
    diff = np.diff(filtered)
    squared = diff ** 2
    win_size = max(1, int(0.12 * fs))
    kernel = np.ones(win_size) / win_size
    integrated = np.convolve(squared, kernel, mode="same")
    min_distance = int(0.3 * fs)
    threshold = np.mean(integrated) + 0.3 * np.std(integrated)
    peaks, _ = find_peaks(integrated, height=threshold, distance=min_distance)
    return peaks


def detect_r_peaks_multichannel(ecg, fs=128):
    best_peaks = np.array([], dtype=int)
    for ch in range(ecg.shape[1]):
        peaks = detect_r_peaks(ecg[:, ch], fs=fs)
        if len(peaks) > len(best_peaks):
            best_peaks = peaks
    return best_peaks


def segment_beats(ecg, r_peaks, fs=128, beat_samples=128):
    half = beat_samples // 2
    n_channels = ecg.shape[1]
    valid_beats = []
    valid_peak_indices = []
    for peak in r_peaks:
        start = peak - half
        end = peak + half
        if start < 0 or end > ecg.shape[0]:
            continue
        valid_beats.append(ecg[start:end, :])
        valid_peak_indices.append(peak)
    if len(valid_beats) == 0:
        return np.zeros((0, beat_samples, n_channels), dtype=np.float32), np.array([], dtype=int)
    beats = np.stack(valid_beats, axis=0)
    return beats, np.array(valid_peak_indices, dtype=int)


def compute_signal_quality(beat):
    if beat.shape[0] == 0:
        return 0.0
    amplitude_range = np.ptp(beat, axis=0).mean()
    if amplitude_range < 5.0:
        return 0.0
    if amplitude_range > 200.0:
        return 0.0
    baseline_drift = np.abs(beat[0].mean() - beat[-1].mean())
    if baseline_drift > 50.0:
        return 0.3
    return 1.0


def beats_to_windows(beats, peak_indices, fs=128, window_sec=300):
    if len(beats) == 0:
        return []
    window_samples = window_sec * fs
    total_duration_samples = peak_indices[-1] - peak_indices[0] if len(peak_indices) > 1 else 0
    n_windows = max(1, int(np.ceil(total_duration_samples / window_samples)))
    windows = []
    for wi in range(n_windows):
        t_start = peak_indices[0] + wi * window_samples
        t_end = t_start + window_samples
        mask = (peak_indices >= t_start) & (peak_indices < t_end)
        win_beats = beats[mask]
        win_peaks = peak_indices[mask]
        if len(win_beats) < 10:
            continue
        qualities = np.array([compute_signal_quality(b) for b in win_beats])
        time_of_day_sec = t_start / fs
        windows.append({
            "beats": win_beats,
            "peak_indices": win_peaks,
            "window_idx": wi,
            "n_beats": len(win_beats),
            "mean_quality": float(qualities.mean()),
            "time_of_day_sec": time_of_day_sec,
        })
    return windows
