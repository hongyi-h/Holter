"""Heart Rate Variability feature extraction."""
import numpy as np
from scipy.signal import welch


def compute_rr_intervals(r_peaks, fs=128):
    """Convert R-peak indices to RR intervals in ms."""
    if len(r_peaks) < 2:
        return np.array([])
    rr = np.diff(r_peaks) / fs * 1000.0  # ms
    return rr


def time_domain_hrv(rr_intervals):
    """Compute time-domain HRV features."""
    if len(rr_intervals) < 2:
        return {"SDNN": 0.0, "RMSSD": 0.0, "pNN50": 0.0, "mean_rr": 0.0}
    sdnn = float(np.std(rr_intervals, ddof=1))
    diffs = np.diff(rr_intervals)
    rmssd = float(np.sqrt(np.mean(diffs ** 2)))
    pnn50 = float(np.sum(np.abs(diffs) > 50) / len(diffs) * 100)
    return {"SDNN": sdnn, "RMSSD": rmssd, "pNN50": pnn50, "mean_rr": float(np.mean(rr_intervals))}


def frequency_domain_hrv(rr_intervals, fs_rr=4.0):
    """Compute frequency-domain HRV via Welch periodogram."""
    if len(rr_intervals) < 30:
        return {"ULF": 0.0, "VLF": 0.0, "LF": 0.0, "HF": 0.0, "LF_HF_ratio": 0.0}
    # Interpolate to uniform sampling
    t = np.cumsum(rr_intervals) / 1000.0
    t = t - t[0]
    from scipy.interpolate import interp1d
    t_uniform = np.arange(0, t[-1], 1.0 / fs_rr)
    if len(t_uniform) < 30:
        return {"ULF": 0.0, "VLF": 0.0, "LF": 0.0, "HF": 0.0, "LF_HF_ratio": 0.0}
    interp_fn = interp1d(t, rr_intervals, kind="linear", fill_value="extrapolate")
    rr_uniform = interp_fn(t_uniform)

    nperseg = min(256, len(rr_uniform))
    freqs, psd = welch(rr_uniform, fs=fs_rr, nperseg=nperseg)

    ulf = np.trapz(psd[(freqs >= 0.0) & (freqs < 0.003)], freqs[(freqs >= 0.0) & (freqs < 0.003)])
    vlf = np.trapz(psd[(freqs >= 0.003) & (freqs < 0.04)], freqs[(freqs >= 0.003) & (freqs < 0.04)])
    lf = np.trapz(psd[(freqs >= 0.04) & (freqs < 0.15)], freqs[(freqs >= 0.04) & (freqs < 0.15)])
    hf = np.trapz(psd[(freqs >= 0.15) & (freqs < 0.4)], freqs[(freqs >= 0.15) & (freqs < 0.4)])
    ratio = lf / hf if hf > 0 else 0.0

    return {"ULF": float(ulf), "VLF": float(vlf), "LF": float(lf), "HF": float(hf), "LF_HF_ratio": float(ratio)}


def diurnal_hr_profile(r_peaks, fs=128, total_hours=24, n_bins=24):
    """Extract hourly heart rate profile."""
    if len(r_peaks) < 2:
        return np.zeros(n_bins)
    rr = compute_rr_intervals(r_peaks, fs)
    hr = 60000.0 / rr  # bpm
    # Assign each beat to an hour bin
    times_hr = np.cumsum(rr) / 1000.0 / 3600.0  # hours
    profile = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    for i, t in enumerate(times_hr):
        bin_idx = int(t) % n_bins
        profile[bin_idx] += hr[i]
        counts[bin_idx] += 1
    counts[counts == 0] = 1
    return profile / counts


def extract_all_hrv(r_peaks, fs=128):
    """Extract all HRV features from R-peaks."""
    rr = compute_rr_intervals(r_peaks, fs)
    td = time_domain_hrv(rr)
    fd = frequency_domain_hrv(rr)
    td.update(fd)
    return td


def extract_window_hrv(r_peaks, fs=128):
    """Extract HRV for a single window."""
    return extract_all_hrv(r_peaks, fs)
