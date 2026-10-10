"""Stress features, an explainable threshold detector, and episode checks.

Two independent regime detectors are provided so each can act as a sanity check
on the other:

* ``detect_regimes_hmm`` (in :mod:`regimes.hmm`) - a 3-state Gaussian HMM on
  standardised volatility features; data-driven, no labels.
* ``threshold_regimes`` here - fixed quantiles of a transparent stress score;
  the "explain it on a whiteboard" baseline.

Both rank regimes so **0 = calm, last = stressed**. The study then reports the
correlation structure inside each regime and how it flips between them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .hmm import detect_regimes_hmm

# Feature construction: short-rate (front) and belly curvature volatility, in
# basis points per day, from the daily changes of the chosen proxy series.
FRONT_SPREAD = "2y-1y"
BELLY_BUTTERFLY = "fly 2*2y-1y-5y"


def _standardize(df: pd.DataFrame, method: str, min_periods: int) -> pd.DataFrame:
    """Standardise features either cross-sectionally (global) or over time.

    ``expanding`` uses only information available up to each date (no
    look-ahead), so a high-volatility decade cannot swamp later episodes the way
    a single full-sample z-score does.
    """
    if method == "global":
        return (df - df.mean()) / df.std(ddof=0)
    if method == "expanding":
        mean = df.expanding(min_periods=min_periods).mean()
        std = df.expanding(min_periods=min_periods).std(ddof=0)
        return (df - mean) / std
    raise ValueError(f"unknown standardize method: {method!r}")


def build_stress_features(
    spreads: pd.DataFrame,
    window: int = 21,
    standardize: str = "expanding",
    min_periods: int = 250,
    front: str = FRONT_SPREAD,
    belly: str = BELLY_BUTTERFLY,
) -> pd.DataFrame:
    """Rolling volatility features (standardised) used for regime detection.

    Features are the ``window``-day rolling standard deviation of daily changes
    in the front-end calendar spread and in the belly butterfly, each converted
    to a z-score. Default ``standardize="expanding"`` uses only past data
    (no look-ahead); ``"global"`` uses the full sample. Higher = more stressed.
    """
    changes = spreads.diff()
    vol = changes.rolling(window, min_periods=max(5, window // 2)).std()
    feats = pd.DataFrame(
        {
            "vol_front": vol[front],
            "vol_belly": vol[belly],
        }
    ).dropna()
    z = _standardize(feats, standardize, min_periods).dropna()
    z.columns = [f"z_{c}" for c in z.columns]
    return z


def stress_score(z_features: pd.DataFrame) -> pd.Series:
    """A single stress score: the mean of the standardised features."""
    return z_features.mean(axis=1).rename("stress_score")


def threshold_regimes(
    score: pd.Series,
    calm_q: float = 0.50,
    stress_q: float = 0.85,
    min_periods: int = 250,
    n_states: int = 3,
) -> pd.Series:
    """Percentile-threshold regime labels (0 calm, 1 normal, 2 stressed).

    Deliberately simple and fully explainable: each day's stress score is ranked
    against its own history up to that day (expanding percentile, no
    look-ahead); the calm band is the lower ``calm_q`` fraction, the stressed
    band the upper ``1 - stress_q`` fraction, and everything else is normal.
    """
    if n_states != 3:
        raise ValueError("threshold_regimes currently supports n_states=3")
    pct = score.expanding(min_periods=min_periods).rank(pct=True)
    labels = pd.Series(1, index=score.index, name="regime")
    labels[pct <= calm_q] = 0
    labels[pct >= stress_q] = 2
    labels[pct.isna()] = 1
    return labels.astype(int)


def transition_matrix(labels: pd.Series, n_states: int = 3) -> pd.DataFrame:
    """Empirical day-to-day transition matrix (row-normalised counts)."""
    x = labels.to_numpy()
    counts = np.zeros((n_states, n_states))
    for a, b in zip(x[:-1], x[1:], strict=False):
        counts[a, b] += 1
    row = counts.sum(axis=1, keepdims=True)
    probs = counts / np.maximum(row, 1)
    return pd.DataFrame(
        probs, index=[f"from_{i}" for i in range(n_states)],
        columns=[f"to_{i}" for i in range(n_states)],
    )


def regime_summary(labels: pd.Series, score: pd.Series | None = None) -> pd.DataFrame:
    """Per-regime day counts, share of sample, date span and mean stress score."""
    rows = []
    n = len(labels)
    for state in sorted(labels.unique()):
        mask = labels == state
        row = {
            "regime": int(state),
            "days": int(mask.sum()),
            "share_pct": round(100.0 * mask.sum() / n, 1),
            "first_date": labels.index[mask][0].date().isoformat(),
            "last_date": labels.index[mask][-1].date().isoformat(),
        }
        if score is not None:
            row["mean_stress_score"] = round(float(score[mask].mean()), 3)
        rows.append(row)
    return pd.DataFrame(rows).set_index("regime")


def episode_coverage(
    labels: pd.Series,
    episodes: dict,
    stressed_state: int | None = None,
    out_of_scope: set | None = None,
) -> pd.DataFrame:
    """How strongly each known episode is flagged by the regime labels.

    Reports two shares per episode:

    * ``elevated_share_pct`` - share of days in any non-calm state (>= 1);
    * ``stressed_share_pct`` - share of days in the top stress state.

    ``detected`` is True when most of the episode's observed days are elevated
    (a short-rate episode should not be calm). ``in_scope`` is False for
    long-end episodes a short-rate study is not expected to flag.
    """
    if stressed_state is None:
        stressed_state = int(labels.max())
    out_of_scope = out_of_scope or set()
    rows = []
    for name, (start, end) in episodes.items():
        in_window = (labels.index >= pd.Timestamp(start)) & (labels.index <= pd.Timestamp(end))
        window = labels.loc[in_window]
        n = len(window)
        elevated = float((window >= 1).mean()) if n else float("nan")
        stressed = float((window == stressed_state).mean()) if n else float("nan")
        rows.append(
            {
                "episode": name,
                "start": start,
                "end": end,
                "days": n,
                "elevated_share_pct": round(100.0 * elevated, 1) if n else None,
                "stressed_share_pct": round(100.0 * stressed, 1) if n else None,
                "detected": bool(n and elevated > 0.5),
                "in_scope": name not in out_of_scope,
            }
        )
    return pd.DataFrame(rows).set_index("episode")


def fit_hmm_regimes(
    z_features: pd.DataFrame, n_states: int = 3, seed: int = 0
) -> tuple[pd.Series, dict]:
    """Thin pass-through so callers import one regime API surface."""
    return detect_regimes_hmm(z_features, n_states=n_states, seed=seed)
