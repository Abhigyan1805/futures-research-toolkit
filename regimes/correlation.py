"""Correlation tools: rolling pairwise correlations and per-regime matrices.

The study looks at correlation structure two ways, because they answer
different questions:

* **levels** (the spread / butterfly values) capture co-movement of the
  *positions themselves*, which is what a spread trader's P&L is exposed to;
* **daily changes** capture co-movement of *shocks*, which is what risk and
  diversification actually live on.

Correlation can differ sharply between the two, and between regimes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_changes(df: pd.DataFrame) -> pd.DataFrame:
    """First difference (per observation day) of each column."""
    return df.diff()


def rolling_correlation(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    """Rolling Pearson correlation of two series over ``window`` observations."""
    out = a.rolling(window, min_periods=window).corr(b)
    out.name = f"corr({a.name},{b.name})"
    return out


def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Pairwise correlation matrix (pairwise-complete observations)."""
    return df.corr(method=method, min_periods=3)


def regime_correlation_matrices(
    df: pd.DataFrame, labels: pd.Series, method: str = "pearson"
) -> dict:
    """Correlation matrix of ``df`` restricted to the dates of each regime.

    ``labels`` is a Series aligned to ``df.index`` whose values are regime ids.
    Returns ``{regime_id: correlation_matrix}`` skipping regimes with too few
    observations to estimate a matrix.
    """
    labels = labels.reindex(df.index)
    out = {}
    for state in sorted(pd.unique(labels.dropna())):
        mask = labels == state
        sub = df.loc[mask].dropna(how="all")
        if sub.shape[0] < 3:
            continue
        out[state] = correlation_matrix(sub, method=method)
    return out


def mean_offdiag_correlation(mat: pd.DataFrame) -> float:
    """Mean of the pairwise (off-diagonal) correlations in a matrix.

    A single robust scalar for "how correlated is the basket overall": the
    classic measure of co-movement that tends to intensify under stress.
    """
    values = mat.to_numpy()
    iu = np.triu_indices(len(values), k=1)
    return float(np.nanmean(values[iu]))


def corr_stability(mat_a: pd.DataFrame, mat_b: pd.DataFrame) -> float:
    """Mean absolute difference between two correlation matrices on common names.

    A simple scalar for "how much did the correlation structure move" between
    two regimes. 0 = identical, larger = more reorganisation.
    """
    common = mat_a.index.intersection(mat_b.index)
    a = mat_a.loc[common, common].to_numpy()
    b = mat_b.loc[common, common].to_numpy()
    iu = np.triu_indices(len(common), k=1)
    return float(np.mean(np.abs(a[iu] - b[iu])))
