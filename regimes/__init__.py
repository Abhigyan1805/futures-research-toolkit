"""Calendar spread / butterfly correlation and regime study.

This package quantifies how correlation structure among short-rate calendar
spreads and butterflies changes across market regimes, and labels calm versus
stressed periods. Inputs come from the shared free FRED loader (``data``).

Individually listed STIR futures contract histories are not free, so
constant-maturity Treasury yields and policy rates stand in for the short-rate
curve a STIR calendar-spread / butterfly trader actually trades. This proxy is
stated everywhere it matters.

Modules:
    config       - series definitions, spread/butterfly baskets, known episodes
    data         - multi-series curve assembly on top of the shared FRED loader
    spreads      - calendar-spread and butterfly construction in basis points
    correlation  - rolling correlations and per-regime correlation matrices
    hmm          - Gaussian-HMM regime detector, implemented from scratch
    regimes      - stress features, threshold detector, episode checks
    plots        - matplotlib figures (regime timeline, rolling corr, heatmaps)
    excel        - Excel summary workbook
    run          - end-to-end entry point (``python -m regimes.run``)
"""

__all__ = [
    "config",
    "data",
    "spreads",
    "correlation",
    "hmm",
    "regimes",
    "plots",
    "excel",
    "run",
]
