"""Curve assembly on top of the shared FRED loader.

The shared ``data.load_fred`` serves one series at a time. This module joins the
constant-maturity tenors into a single dated frame and appends the policy-rate
references, so the rest of the package sees one tidy input.

Data provenance and the proxy caveat
------------------------------------
* Source: free FRED CSV endpoint via the shared loader (no API key).
* Individually listed STIR futures contract histories are *not* free, so
  constant-maturity Treasury yields stand in for the short-rate curve that STIR
  calendar spreads and butterflies reference. See ``docs/regimes/.md``.
"""

from __future__ import annotations

import pandas as pd

from data import load_fred as _shared_load_fred

from . import config


def load_curve(
    tenors=None,
    start: str | None = config.STUDY_START,
    end: str | None = None,
    *,
    offline: bool | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Join constant-maturity Treasury tenors into one dated, float frame.

    Columns are the series ids in ``tenors`` order (default ``config.CURVE_TENORS``).
    """
    tenors = list(tenors) if tenors is not None else list(config.CURVE_TENORS)
    frames = [
        _shared_load_fred(s, start=start, end=end, offline=offline, refresh=refresh)
        for s in tenors
    ]
    df = pd.concat(frames, axis=1).sort_index()
    df.index.name = "date"
    return df[tenors]


def load_policy(
    series=None,
    start: str | None = config.STUDY_START,
    end: str | None = None,
    *,
    offline: bool | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Join policy / short-rate reference series (DFF, SOFR) into one frame."""
    series = list(series) if series is not None else list(config.POLICY_SERIES)
    frames = [
        _shared_load_fred(s, start=start, end=end, offline=offline, refresh=refresh)
        for s in series
    ]
    df = pd.concat(frames, axis=1).sort_index()
    df.index.name = "date"
    return df[[s for s in series if s in df.columns]]


def load_bundle(
    start: str | None = config.STUDY_START,
    end: str | None = None,
    *,
    offline: bool | None = None,
    refresh: bool = False,
) -> dict:
    """Convenience: ``{"curve": DataFrame, "policy": DataFrame}``."""
    return {
        "curve": load_curve(start=start, end=end, offline=offline, refresh=refresh),
        "policy": load_policy(start=start, end=end, offline=offline, refresh=refresh),
    }
