"""Calendar-spread and butterfly construction, in basis points.

A calendar spread is the yield difference between a far and a near tenor
(steepness of a curve segment). A butterfly is the second difference of the
curve: ``2*mid - near - far`` (curvature of a segment). Both are expressed in
basis points (1 bp = 0.01 percentage point) because yields arrive in percent.

These are *yield-curve* proxies for the STIR calendar spreads and butterflies a
rates trader quotes; see docs/regimes/.md for the proxy caveat.
"""

from __future__ import annotations

import pandas as pd

from . import config

BP = 100.0  # percent -> basis points


def calendar_spread(levels: pd.DataFrame, near: str, far: str, scale: float = BP) -> pd.Series:
    """Far-minus-near yield in basis points."""
    s = (levels[far] - levels[near]) * scale
    s.name = config.spread_name(near, far)
    return s


def butterfly(levels: pd.DataFrame, near: str, mid: str, far: str, scale: float = BP) -> pd.Series:
    """``2*mid - near - far`` curvature in basis points."""
    s = (2.0 * levels[mid] - levels[near] - levels[far]) * scale
    s.name = config.butterfly_name(near, mid, far)
    return s


def build_calendar_spreads(levels: pd.DataFrame, pairs=None, scale: float = BP) -> pd.DataFrame:
    """DataFrame of calendar spreads (one column per pair)."""
    pairs = pairs if pairs is not None else config.CALENDAR_PAIRS
    cols = [calendar_spread(levels, n, f, scale) for n, f in pairs]
    return pd.concat(cols, axis=1)


def build_butterflies(levels: pd.DataFrame, triples=None, scale: float = BP) -> pd.DataFrame:
    """DataFrame of butterfly spreads (one column per triple)."""
    triples = triples if triples is not None else config.BUTTERFLY_TRIPLES
    cols = [butterfly(levels, n, m, f, scale) for n, m, f in triples]
    return pd.concat(cols, axis=1)


def build_spread_basket(levels: pd.DataFrame, scale: float = BP) -> pd.DataFrame:
    """Calendar spreads and butterflies stacked into one basket of proxies."""
    spreads = build_calendar_spreads(levels, scale=scale)
    flies = build_butterflies(levels, scale=scale)
    return pd.concat([spreads, flies], axis=1)
