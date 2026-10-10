"""Statistical significance of candlestick patterns.

For every pattern and every forward horizon (1, 5 and 10 trading days) we ask:
*does the pattern carry information about the next move, beyond chance?*  We
answer with three complementary tests and correct for multiple comparisons.

Method
------
1. **Directional hit rate vs base rate** (binomial).  A pattern with a
   traditional direction ("bullish"/"bearish") "hits" when the forward return
   has the same sign.  Under the null the hit probability is the *unconditional*
   probability of that move direction (the base rate), so we use an exact
   two-sided binomial test.  This is the primary test.
2. **One-sample t-test** of the pattern's mean forward return against zero.
3. **Random-entry bootstrap.**  Draw the same number of entries at random dates
   (with replacement) and record the mean forward return; repeat ``n_boot``
   times.  The bootstrap p-value is how often a random-entry strategy matches or
   beats the pattern -- a "matched random baseline" that controls for the market
   simply drifting up.

Multiple testing
----------------
Running ~14 patterns x 3 horizons = 42 tests guarantees false positives at 5%,
so the primary p-values are passed through the Benjamini-Hochberg step-up
procedure and reported as ``primary_p_bh`` with a ``bh_significant`` flag.

Honesty
-------
Most textbook candlestick patterns do **not** beat chance in a large liquid
futures sample.  The report says so, counts how many survive BH correction, and
names them; no cherry-picking.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# exact binomial test (two-sided, method of small p-values) -- no scipy needed
# ---------------------------------------------------------------------------


def binom_test(k: int, n: int, p: float) -> float:
    """Exact two-sided binomial p-value for ``k`` successes in ``n`` trials."""
    if n <= 0:
        return float("nan")
    p = min(max(p, 0.0), 1.0)
    if not (0 <= k <= n):
        raise ValueError("k must satisfy 0 <= k <= n")
    if p == 0.0:
        return 0.0 if k > 0 else 1.0
    # log pmf to avoid overflow for large n
    logs = np.array([_log_comb(n, i) + i * math.log(p) + (n - i) * math.log1p(-p)
                     if p < 1.0 else (0.0 if i == n else -np.inf)
                     for i in range(n + 1)])
    pmf = np.exp(logs - logs.max())
    observed = pmf[k]
    total = pmf[pmf <= observed * (1.0 + 1e-9)].sum()
    return float(min(1.0, total / pmf.sum()))


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


# ---------------------------------------------------------------------------
# one-sample t-test (Student) via the regularised incomplete beta function
# ---------------------------------------------------------------------------


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Numerical Recipes)."""
    maxit, eps, fpmin = 300, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def ttest_1samp(sample: np.ndarray, popmean: float = 0.0) -> tuple[float, float]:
    """Two-sided one-sample Student t-test; returns ``(t, p)``."""
    x = np.asarray(sample, dtype=float)
    n = x.size
    if n < 2:
        return float("nan"), float("nan")
    mean = x.mean()
    sd = x.std(ddof=1)
    if sd == 0.0:
        return (float("inf"), 0.0) if mean != popmean else (0.0, float("nan"))
    t = (mean - popmean) / (sd / math.sqrt(n))
    df = n - 1
    p = _betai(df / 2.0, 0.5, df / (df + t * t))
    return float(t), float(p)


# ---------------------------------------------------------------------------
# Benjamini-Hochberg multiple-testing correction
# ---------------------------------------------------------------------------


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Return BH-adjusted p-values (same order as input, NaNs preserved)."""
    p = np.asarray(pvals, dtype=float)
    adjusted = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return adjusted
    order = np.argsort(p[ok])
    ranked = p[ok][order]
    m_arr = np.arange(1, m + 1)
    adj = ranked * m / m_arr
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out = np.empty(m)
    out[order] = adj
    adjusted[ok] = out
    return adjusted


# ---------------------------------------------------------------------------
# forward returns and event statistics
# ---------------------------------------------------------------------------


def forward_returns(close: pd.Series, horizons: tuple[int, ...] = (1, 5, 10)) -> pd.DataFrame:
    """Simple close-to-close forward returns for each horizon (no look-ahead bent)."""
    out = {}
    for h in horizons:
        out[f"fwd_ret_{h}"] = close.shift(-h) / close - 1.0
    return pd.DataFrame(out, index=close.index)


def _bootstrap_p(event_returns: np.ndarray, universe_returns: np.ndarray,
                 n_boot: int, rng: np.random.Generator, *, bullish: bool) -> float:
    """Random-entry baseline p-value for the pattern's mean forward return."""
    event_returns = event_returns[np.isfinite(event_returns)]
    universe = universe_returns[np.isfinite(universe_returns)]
    n = event_returns.size
    if n < 2 or universe.size < n:
        return float("nan")
    observed = event_returns.mean()
    draws = rng.choice(universe, size=(n_boot, n), replace=True)
    means = draws.mean(axis=1)
    if bullish:
        return float((np.sum(means >= observed) + 1) / (n_boot + 1))
    return float((np.sum(means <= observed) + 1) / (n_boot + 1))


def evaluate(candles: pd.DataFrame, patterns: tuple[str, ...],
             sides: dict[str, str], horizons: tuple[int, ...] = (1, 5, 10),
             *, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> pd.DataFrame:
    """Run every pattern x horizon test and return one tidy summary row each.

    ``candles`` must already contain the boolean pattern columns and the
    ``fwd_ret_<h>`` columns (see :func:`candles.pipeline.build_candles`).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for name in patterns:
        side = sides.get(name, "neutral")
        signal = candles[name].to_numpy(dtype=bool)
        for h in horizons:
            col = f"fwd_ret_{h}"
            ret_all = candles[col].to_numpy(dtype=float)
            event = ret_all[signal]
            event = event[np.isfinite(event)]
            universe = ret_all[np.isfinite(ret_all)]
            n = event.size
            row = {
                "pattern": name,
                "side": side,
                "horizon": h,
                "n_signals": int(n),
            }
            if n == 0:
                row.update({"hit_rate": np.nan, "base_rate": np.nan, "binom_p": np.nan,
                            "mean_return": np.nan, "median_return": np.nan, "t_stat": np.nan,
                            "t_p": np.nan, "boot_p": np.nan})
                rows.append(row)
                continue
            up = universe > 0
            if side == "bullish":
                hits = int((event > 0).sum())
                base = float(up.mean())
            elif side == "bearish":
                hits = int((event < 0).sum())
                base = float((~up).mean())
            else:
                hits, base = None, float("nan")
            row["mean_return"] = float(event.mean())
            row["median_return"] = float(np.median(event))
            if hits is not None:
                row["hit_rate"] = hits / n
                row["base_rate"] = base
                row["binom_p"] = binom_test(hits, n, base)
            else:
                row["hit_rate"] = np.nan
                row["base_rate"] = np.nan
                row["binom_p"] = np.nan
            t, p = ttest_1samp(event, 0.0)
            row["t_stat"] = t
            row["t_p"] = p
            row["boot_p"] = _bootstrap_p(event, universe, n_boot, rng,
                                         bullish=(side != "bearish"))
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary["primary_p"] = summary["binom_p"].where(summary["binom_p"].notna(),
                                                    summary["t_p"])
    summary["primary_p_bh"] = benjamini_hochberg(summary["primary_p"].to_numpy())
    summary["bh_significant"] = summary["primary_p_bh"] <= alpha
    return summary
