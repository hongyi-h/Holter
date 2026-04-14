"""Circadian Electrical Reserve (CER) metric."""
import numpy as np
from sklearn.decomposition import PCA


def compute_cer(window_embeddings, quality_mask):
    """
    Compute CER = H1 * (1 - F) from window embeddings.
    
    Args:
        window_embeddings: (W, D) array of window embeddings
        quality_mask: (W,) boolean array
    Returns:
        dict with H1, F, CER, n_valid, pca_explained_var, status
    """
    if quality_mask is not None:
        valid = window_embeddings[quality_mask]
    else:
        valid = window_embeddings

    n_valid = len(valid)
    if n_valid < 10:
        return {"H1": 0.0, "F": 1.0, "CER": 0.0, "n_valid": n_valid,
                "pca_explained_var": 0.0, "status": "insufficient_data"}

    # PCA to reduce dimensions
    n_components = min(10, n_valid, valid.shape[1])
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(valid)
    explained_var = pca.explained_variance_ratio_.sum()

    # H1: Spectral entropy of FFT magnitudes (first PC)
    pc1 = projected[:, 0]
    fft_mag = np.abs(np.fft.rfft(pc1 - pc1.mean()))
    fft_mag = fft_mag[1:]  # Remove DC
    if fft_mag.sum() == 0:
        return {"H1": 0.0, "F": 1.0, "CER": 0.0, "n_valid": n_valid,
                "pca_explained_var": explained_var, "status": "flat_signal"}
    
    psd = fft_mag ** 2
    psd_norm = psd / psd.sum()
    psd_norm = psd_norm[psd_norm > 0]
    H1 = -np.sum(psd_norm * np.log2(psd_norm)) / np.log2(len(psd_norm)) if len(psd_norm) > 1 else 0.0

    # F: Flatness ratio (geometric mean / arithmetic mean of PSD)
    geo_mean = np.exp(np.mean(np.log(psd + 1e-12)))
    arith_mean = np.mean(psd)
    F = geo_mean / (arith_mean + 1e-12)
    F = np.clip(F, 0.0, 1.0)

    CER = H1 * (1.0 - F)

    return {
        "H1": float(H1),
        "F": float(F),
        "CER": float(CER),
        "n_valid": n_valid,
        "pca_explained_var": float(explained_var),
        "status": "ok",
    }


def compute_cer_batch(all_embeddings, quality_masks):
    """Compute CER for multiple patients."""
    results = {}
    for pid, emb in all_embeddings.items():
        qm = quality_masks.get(pid, np.ones(emb.shape[0], dtype=bool))
        results[pid] = compute_cer(emb, qm)
    return results


def compute_split_half_icc(window_embeddings, quality_mask, n_bootstrap=100):
    """Split-half ICC for CER reliability."""
    if quality_mask is not None:
        valid = window_embeddings[quality_mask]
    else:
        valid = window_embeddings

    if len(valid) < 20:
        return 0.0

    cer_halves = []
    for _ in range(n_bootstrap):
        idx = np.random.permutation(len(valid))
        half1 = valid[idx[:len(idx)//2]]
        half2 = valid[idx[len(idx)//2:]]
        m1 = np.ones(len(half1), dtype=bool)
        m2 = np.ones(len(half2), dtype=bool)
        c1 = compute_cer(half1, m1)["CER"]
        c2 = compute_cer(half2, m2)["CER"]
        cer_halves.append((c1, c2))

    cer_halves = np.array(cer_halves)
    if cer_halves.std(axis=0).sum() < 1e-10:
        return 1.0
    corr = np.corrcoef(cer_halves[:, 0], cer_halves[:, 1])[0, 1]
    icc = 2 * corr / (1 + corr) if corr > 0 else 0.0
    return float(icc)


def compute_circadian_band_energy(window_embeddings, quality_mask, sample_rate_cpd=None):
    """
    Compute fraction of spectral energy in the circadian band (0.8-1.2 cpd)
    and ultradian band (2-6 cpd). Windows are assumed to be ~5-min each,
    so 288 windows = 1 day => sample_rate = 288 cpd.
    
    Returns dict with circadian_energy, ultradian_energy, circ_ultra_ratio.
    """
    if quality_mask is not None:
        valid = window_embeddings[quality_mask]
    else:
        valid = window_embeddings

    n = len(valid)
    if n < 20:
        return {'circadian_energy': 0.0, 'ultradian_energy': 0.0,
                'circ_ultra_ratio': 0.0, 'status': 'insufficient_data'}

    if sample_rate_cpd is None:
        sample_rate_cpd = float(n)  # assume 1 day of data

    # PCA project to first 5 PCs
    from sklearn.decomposition import PCA
    n_comp = min(5, n, valid.shape[1])
    pca = PCA(n_components=n_comp)
    projected = pca.fit_transform(valid)

    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_cpd)
    total_energy = 0.0
    circ_energy = 0.0
    ultra_energy = 0.0

    for pc_i in range(n_comp):
        fft_mag = np.abs(np.fft.rfft(projected[:, pc_i] - projected[:, pc_i].mean()))
        psd = fft_mag ** 2
        # Weight by explained variance
        w = pca.explained_variance_ratio_[pc_i]
        total_energy += w * psd[1:].sum()
        circ_mask = (freqs[1:] >= 0.8) & (freqs[1:] <= 1.2)
        ultra_mask = (freqs[1:] >= 2.0) & (freqs[1:] <= 6.0)
        circ_energy += w * psd[1:][circ_mask].sum()
        ultra_energy += w * psd[1:][ultra_mask].sum()

    if total_energy < 1e-12:
        return {'circadian_energy': 0.0, 'ultradian_energy': 0.0,
                'circ_ultra_ratio': 0.0, 'status': 'flat_signal'}

    return {
        'circadian_energy': float(circ_energy / total_energy),
        'ultradian_energy': float(ultra_energy / total_energy),
        'circ_ultra_ratio': float(circ_energy / (ultra_energy + 1e-12)),
        'status': 'ok',
    }

