"""Evaluation utilities: AUROC, linear probe, CI estimation."""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score


def auroc_safe(y_true, y_score):
    """AUROC with fallback for edge cases."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def auprc_safe(y_true, y_score):
    """AUPRC with fallback."""
    if len(np.unique(y_true)) < 2:
        return float(np.mean(y_true))
    return float(average_precision_score(y_true, y_score))


def bootstrap_auroc_ci(y_true, y_score, n_boot=1000, alpha=0.05):
    """Bootstrap 95% CI for AUROC."""
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = np.random.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    if len(aucs) == 0:
        return {"mean": 0.5, "ci_low": 0.5, "ci_high": 0.5}
    aucs = np.array(aucs)
    return {
        "mean": float(aucs.mean()),
        "ci_low": float(np.percentile(aucs, 100 * alpha / 2)),
        "ci_high": float(np.percentile(aucs, 100 * (1 - alpha / 2))),
    }


def bootstrap_delta_auroc(y_true, y_score1, y_score2, n_boot=1000):
    """Bootstrap test for AUROC difference."""
    deltas = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = np.random.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        a1 = roc_auc_score(y_true[idx], y_score1[idx])
        a2 = roc_auc_score(y_true[idx], y_score2[idx])
        deltas.append(a1 - a2)
    if len(deltas) == 0:
        return {"delta_mean": 0.0, "p_value": 1.0}
    deltas = np.array(deltas)
    p_val = np.mean(deltas <= 0)
    return {"delta_mean": float(deltas.mean()), "p_value": float(p_val)}


def partial_r_squared(y, x_full, x_reduced):
    """Partial R-squared: how much variance x_full explains beyond x_reduced."""
    from sklearn.linear_model import LinearRegression
    r2_full = LinearRegression().fit(x_full, y).score(x_full, y)
    r2_reduced = LinearRegression().fit(x_reduced, y).score(x_reduced, y)
    return r2_full - r2_reduced


def linear_probe_cv(X, y, n_folds=5):
    """5-fold CV logistic regression linear probe."""
    if len(np.unique(y)) < 2:
        return {"auroc": 0.5, "auroc_ci": {"mean": 0.5, "ci_low": 0.5, "ci_high": 0.5},
                "auprc": 0.5, "fold_aucs": [0.5]}

    skf = StratifiedKFold(n_splits=min(n_folds, len(y) // 2), shuffle=True, random_state=42)
    fold_aucs = []
    fold_auprcs = []
    all_scores = np.zeros(len(y))

    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
        clf.fit(X[train_idx], y[train_idx])
        scores = clf.predict_proba(X[test_idx])
        if scores.shape[1] == 2:
            scores = scores[:, 1]
        else:
            scores = scores[:, 0]
        all_scores[test_idx] = scores
        fold_aucs.append(auroc_safe(y[test_idx], scores))
        fold_auprcs.append(auprc_safe(y[test_idx], scores))

    ci = bootstrap_auroc_ci(y, all_scores)
    return {
        "auroc": float(np.mean(fold_aucs)),
        "auroc_ci": ci,
        "auprc": float(np.mean(fold_auprcs)),
        "fold_aucs": fold_aucs,
    }


def save_results(results, path):
    """Save results dict as JSON."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
