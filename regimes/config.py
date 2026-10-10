"""Central configuration for the regime study.

Everything that defines *what* is measured lives here so it can be wired to the
shared contract (``docs/CONTRACTS.md``) without touching the logic modules.

Series set is fixed to the shared FRED loader's supported ids
(``DGS3MO DGS1 DGS2 DGS5 DGS10 SOFR DFF``) so offline fixtures and CI work.
"""

from __future__ import annotations

# Constant-maturity Treasury yields used to build the short-rate curve.
# Constant-maturity for *all* tenors keeps 3m comparable with 1y+; the
# discount-basis T-bill series (DTB3/DTB6) are not mixed in (see docs/regimes/.md).
CURVE_TENORS = ["DGS3MO", "DGS1", "DGS2", "DGS5", "DGS10"]

# Policy / short-rate reference series (context, not part of the curve basket).
POLICY_SERIES = ["DFF", "SOFR"]

ALL_SERIES = CURVE_TENORS + POLICY_SERIES

# Calendar-spread proxies: (near leg, far leg). Positive value => that segment
# is upward sloping (far yield above near yield).
CALENDAR_PAIRS = [
    ("DGS3MO", "DGS1"),   # 1y-3m  (front-end steepness)
    ("DGS1", "DGS2"),     # 2y-1y
    ("DGS2", "DGS5"),     # 5y-2y
    ("DGS5", "DGS10"),    # 10y-5y
    ("DGS1", "DGS5"),     # 5y-1y
]

# Butterfly proxies: (near, mid, far) with weights 2*mid - near - far (classic
# body/wing STIR butterfly shape, curvature of the segment).
BUTTERFLY_TRIPLES = [
    ("DGS3MO", "DGS1", "DGS2"),    # 2*1y - 3m - 2y
    ("DGS1", "DGS2", "DGS5"),      # 2*2y - 1y - 5y
    ("DGS2", "DGS5", "DGS10"),     # 2*5y - 2y - 10y
]

# Human-readable labels (short) for plots and the Excel workbook.
LEG_LABELS = {
    "DGS3MO": "3m",
    "DGS1": "1y",
    "DGS2": "2y",
    "DGS5": "5y",
    "DGS10": "10y",
    "DFF": "DFF",
    "SOFR": "SOFR",
}


def spread_name(near: str, far: str) -> str:
    """e.g. ``('DGS1','DGS2') -> '2y-1y'``."""
    return f"{LEG_LABELS[far]}-{LEG_LABELS[near]}"


def butterfly_name(near: str, mid: str, far: str) -> str:
    """e.g. ``('DGS1','DGS2','DGS5') -> 'fly 2*2y-1y-5y'``."""
    return f"fly 2*{LEG_LABELS[mid]}-{LEG_LABELS[near]}-{LEG_LABELS[far]}"


# ---------------------------------------------------------------------------
# Known stress episodes used as an honesty check on the regime labels. A useful
# detector should flag most of these windows as stressed.
# ---------------------------------------------------------------------------
KNOWN_EPISODES = {
    "GFC 2008": ("2008-09-01", "2009-06-30"),
    "COVID 2020": ("2020-02-15", "2020-06-30"),
    "Hiking cycle 2022": ("2022-01-01", "2023-12-31"),
    "Taper tantrum 2013": ("2013-05-01", "2013-09-30"),
}

# Constant-maturity 3m begins 1981-09; start at a clean year edge.
STUDY_START = "1982-01-01"

# Regime model defaults (seed fixed so reruns reproduce).
N_STATES = 3
FEATURE_WINDOW = 21      # trading days for rolling volatility features (~1 month)
RANDOM_SEED = 0
