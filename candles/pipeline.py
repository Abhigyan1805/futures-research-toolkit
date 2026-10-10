"""Data plumbing: label a universe of instruments and pool the event study.

The :func:`build_candles` helper turns one OHLC frame into a labelled frame with
forward returns; :func:`evaluate_universe` concatenates several instruments
(keeping each instrument's forward returns inside its own calendar) and runs the
significance tests on the pooled event sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from candles.config import DEFAULT_CONFIG, PATTERN_NAMES, PATTERN_SIDES, PatternConfig
from candles.patterns import label
from candles.significance import evaluate, forward_returns

DEFAULT_TICKERS: tuple[str, ...] = ("ES=F", "NQ=F", "GC=F", "CL=F", "ZN=F")
HORIZONS: tuple[int, ...] = (1, 5, 10)


def build_candles(ohlc: pd.DataFrame, cfg: PatternConfig = DEFAULT_CONFIG,
                  horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """Label every candle and append forward returns for each horizon."""
    labelled = label(ohlc, cfg)
    fwd = forward_returns(ohlc["close"], horizons)
    return pd.concat([labelled, fwd], axis=1)


def evaluate_universe(frames: dict[str, pd.DataFrame],
                      cfg: PatternConfig = DEFAULT_CONFIG,
                      horizons: tuple[int, ...] = HORIZONS,
                      *, seed: int = 0, n_boot: int = 2000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pool labelled candles across instruments and test each pattern x horizon.

    Returns ``(pooled_candles, summary)``.  Forward returns are computed per
    instrument *before* concatenation so no return spans a roll or a weekend gap
    between instruments.
    """
    combined = []
    for ticker, df in frames.items():
        labelled = build_candles(df, cfg, horizons)
        labelled.insert(0, "ticker", ticker)
        combined.append(labelled)
    if not combined:
        raise ValueError("evaluate_universe received no instruments")
    pooled = pd.concat(combined, axis=0)
    summary = evaluate(pooled, PATTERN_NAMES, PATTERN_SIDES, horizons,
                       seed=seed, n_boot=n_boot)
    return pooled, summary


def ticker_event_counts(pooled: pd.DataFrame,
                        patterns: tuple[str, ...] = PATTERN_NAMES) -> pd.DataFrame:
    """Per-instrument signal counts, for the honesty table in the results."""
    rows = []
    for ticker, grp in pooled.groupby("ticker", sort=True):
        row = {"ticker": ticker, "n_bars": int(len(grp))}
        for name in patterns:
            row[name] = int(grp[name].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _finite_mean(x: pd.Series) -> float:
    vals = x.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else float("nan")
