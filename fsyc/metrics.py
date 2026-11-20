"""Scoring rules and calibration metrics for binary probability forecasts.

All functions take p (forecast probability of YES) and y (outcome, 1=YES, 0=NO)
as array-likes of equal length.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-6


def _arr(p, y):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if p.shape != y.shape:
        raise ValueError(f"shape mismatch {p.shape} vs {y.shape}")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be in [0, 1]")
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("outcomes must be 0 or 1")
    return p, y


def brier(p, y) -> float:
    p, y = _arr(p, y)
    return float(np.mean((p - y) ** 2))


def log_loss(p, y) -> float:
    p, y = _arr(p, y)
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability_bins(p, y, n_bins: int = 10):
    """Equal-width bins. Returns list of dicts with count, mean forecast, observed freq."""
    p, y = _arr(p, y)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        out.append({
            "bin_lo": float(edges[b]),
            "bin_hi": float(edges[b + 1]),
            "n": n,
            "mean_p": float(p[m].mean()) if n else np.nan,
            "freq_y": float(y[m].mean()) if n else np.nan,
        })
    return out


def ece(p, y, n_bins: int = 10) -> float:
    """Expected calibration error: count-weighted |mean forecast - observed freq|."""
    p, _ = _arr(p, y)
    bins = reliability_bins(p, y, n_bins)
    total = len(p)
    return float(sum(b["n"] / total * abs(b["mean_p"] - b["freq_y"]) for b in bins if b["n"]))


def murphy_decomposition(p, y, n_bins: int = 10) -> dict:
    """Binned Murphy decomposition: Brier ~= reliability - resolution + uncertainty.

    Exact only when forecasts are constant within each bin; the residual is
    reported so you can see how much binning costs. Reliability is the
    calibration penalty (lower is better); resolution is the ability to
    separate YES from NO cases (higher is better) -- the signal-detection part.
    """
    p, y = _arr(p, y)
    n = len(p)
    base = y.mean()
    rel = res = 0.0
    for b in reliability_bins(p, y, n_bins):
        if b["n"]:
            w = b["n"] / n
            rel += w * (b["mean_p"] - b["freq_y"]) ** 2
            res += w * (b["freq_y"] - base) ** 2
    unc = base * (1 - base)
    bs = brier(p, y)
    return {"brier": bs, "reliability": rel, "resolution": res,
            "uncertainty": unc, "residual": bs - (rel - res + unc)}


def auroc(p, y) -> float:
    """Discrimination only (ignores calibration). Mann-Whitney U with tie handling."""
    p, y = _arr(p, y)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def bootstrap_ci(stat, p, y, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI over questions for any stat(p, y)."""
    p, y = _arr(p, y)
    rng = np.random.default_rng(seed)
    n = len(p)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        v = stat(p[i], y[i])
        if not np.isnan(v):
            vals.append(v)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(stat(p, y)), float(lo), float(hi)


def paired_bootstrap_diff(stat, p_a, p_b, y, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """CI for stat(p_b, y) - stat(p_a, y), resampling the same questions for both.

    Use this to test whether a perturbation (b) changed a metric relative to
    baseline (a). Pairing matters: the questions are shared, so unpaired CIs
    overstate the uncertainty of the difference.
    """
    p_a, y = _arr(p_a, y)
    p_b, _ = _arr(p_b, y)
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        diffs.append(stat(p_b[i], y[i]) - stat(p_a[i], y[i]))
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(stat(p_b, y) - stat(p_a, y)), float(lo), float(hi)


def mean_shift(p_base, p_pert, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Mean of (p_pert - p_base) with a bootstrap CI. Outcome-free: measures
    how far a perturbation moves the forecast, e.g. sycophancy toward a
    user's stated belief."""
    a = np.asarray(p_base, float)
    b = np.asarray(p_pert, float)
    d = b - a
    rng = np.random.default_rng(seed)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)
