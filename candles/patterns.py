"""Vectorised candlestick pattern detectors.

The engine works on a pandas DataFrame of daily OHLC bars (columns
``open, high, low, close``, lowercase, as produced by :func:`data.load_ohlc`).
Every candle is classified independently of its neighbours except for the trend
context and the multi-candle patterns, which look back one or two bars.

Design choices
--------------
* **Relative, not absolute, thresholds.**  A hammer is defined by its *shape*
  (long lower shadow relative to its own body/range), so the same rules work
  across ES (prices ~5000), GC (~2000) and ZN (~110).  We never hard-code price
  distances.
* **Explicit trend context.**  Reversal candles (hammer, shooting star, stars)
  only count when the preceding ``trend_lookback`` bars actually moved the
  "wrong" way; continuation candles (soldiers/crows) do not need it.  Set
  ``require_trend_*`` to ``False`` in :class:`candles.config.PatternConfig` to
  disable this.
* **Vectorised.**  All detectors are boolean numpy operations over shifted
  columns -- no Python loop over candles -- so 10+ years x several instruments
  is instant.
* **No look-ahead.**  A pattern at bar ``i`` may use bars ``i``, ``i-1`` and
  ``i-2`` only.  Forward returns are computed later, in
  :mod:`candles.significance`, so detection never peeks at the future.

The rules for each pattern are stated in the docstring of its detector and in
``docs/candles/index.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from candles.config import DEFAULT_CONFIG, PATTERN_NAMES, PatternConfig

OHLC_COLS = ("open", "high", "low", "close")


class _Features:
    """Per-bar shape quantities shared by every detector."""

    __slots__ = ("open", "high", "low", "close", "rng", "body", "upper", "lower",
                 "body_top", "body_bot", "body_eff", "bull", "bear", "prior_ret")

    def __init__(self, ohlc: pd.DataFrame, cfg: PatternConfig) -> None:
        missing = [c for c in OHLC_COLS if c not in ohlc.columns]
        if missing:
            raise ValueError(f"OHLC frame is missing columns: {missing}")
        o = ohlc["open"].to_numpy(dtype=float)
        h = ohlc["high"].to_numpy(dtype=float)
        lo = ohlc["low"].to_numpy(dtype=float)
        c = ohlc["close"].to_numpy(dtype=float)
        self.open, self.high, self.low, self.close = o, h, lo, c
        self.rng = h - lo
        self.body = np.abs(c - o)
        self.body_top = np.maximum(o, c)
        self.body_bot = np.minimum(o, c)
        self.upper = h - self.body_top
        self.lower = self.body_bot - lo
        # body floored relative to range so near-doji bars do not blow up ratios
        self.body_eff = np.maximum(self.body, cfg.body_floor * self.rng)
        self.bull = c > o
        self.bear = c < o
        # prior return over the trend_lookback bars ending at the previous bar
        prev = pd.Series(c, index=ohlc.index).shift(1)
        base = pd.Series(c, index=ohlc.index).shift(1 + cfg.trend_lookback)
        self.prior_ret = (prev / base - 1.0).to_numpy(dtype=float)


def _shift_bool(a: np.ndarray, n: int) -> np.ndarray:
    """Return a shifted boolean array with False in the newly exposed head."""
    out = np.zeros_like(a, dtype=bool)
    if n > 0:
        out[n:] = a[:-n]
    elif n < 0:
        out[:n] = a[-n:]
    else:
        out = a.copy()
    return out


def detect(ohlc: pd.DataFrame, cfg: PatternConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Return a boolean DataFrame, one column per pattern, aligned to ``ohlc``.

    A ``True`` cell means "bar *i* is this pattern".  A bar can carry several
    patterns at once (e.g. a doji can also be a marubozu only in degenerate
    cases, but a doji inside an engulfing is common); the caller decides how to
    treat overlaps.
    """
    f = _Features(ohlc, cfg)
    valid = f.rng > 0.0
    rng = f.rng
    body = f.body
    body_eff = np.where(f.body_eff > 0, f.body_eff, np.inf)

    trend_down = f.prior_ret < -cfg.trend_min_move
    trend_up = f.prior_ret > cfg.trend_min_move
    hammer_trend = trend_down if cfg.require_trend_hammer else valid

    # ---- single-candle patterns -------------------------------------------
    doji = valid & (body <= cfg.doji_body_max * rng)

    hammer_shape = (
        valid
        & (body <= cfg.small_body_max * rng)
        & (f.lower >= cfg.hammer_shadow_min * body_eff)
        & (f.upper <= cfg.hammer_opp_shadow_max * rng)
    )
    hammer = hammer_shape & hammer_trend

    inv_hammer_shape = (
        valid
        & (body <= cfg.small_body_max * rng)
        & (f.upper >= cfg.star_shadow_min * body_eff)
        & (f.lower <= cfg.star_opp_shadow_max * rng)
    )
    inverted_hammer = (
        (inv_hammer_shape & trend_down) if cfg.require_trend_hammer else inv_hammer_shape
    )
    shooting_star = (
        (inv_hammer_shape & trend_up) if cfg.require_trend_star else inv_hammer_shape
    )

    marubozu_shape = (
        valid
        & (body >= cfg.marubozu_body_min * rng)
        & (f.upper <= cfg.marubozu_shadow_max * rng)
        & (f.lower <= cfg.marubozu_shadow_max * rng)
    )
    marubozu_bullish = marubozu_shape & f.bull
    marubozu_bearish = marubozu_shape & f.bear

    # ---- double-candle patterns -------------------------------------------
    o, c = f.open, f.close
    body_top, body_bot = f.body_top, f.body_bot
    body_prev = _shift_bool(body > 0, 1)
    prev_bear = _shift_bool(f.bear, 1)
    prev_bull = _shift_bool(f.bull, 1)
    prev_body_top = _shift_at(body_top, 1)
    prev_body_bot = _shift_at(body_bot, 1)
    prev_body = _shift_at(body, 1)
    prev_rng = _shift_at(rng, 1)

    bullish_engulfing = (
        valid
        & prev_bear
        & f.bull
        & (c > prev_body_top)
        & (o < prev_body_bot)
        & (prev_body >= cfg.engulf_prior_body_min * prev_rng)
    )
    bearish_engulfing = (
        valid
        & prev_bull
        & f.bear
        & (c < prev_body_bot)
        & (o > prev_body_top)
        & (prev_body >= cfg.engulf_prior_body_min * prev_rng)
    )

    inside_prev = (body_top < prev_body_top) & (body_bot > prev_body_bot)
    small_vs_prev = body <= cfg.harami_body_max * prev_body
    bullish_harami = valid & prev_bear & f.bull & inside_prev & small_vs_prev & body_prev
    bearish_harami = valid & prev_bull & f.bear & inside_prev & small_vs_prev & body_prev

    # ---- triple-candle patterns -------------------------------------------
    avg_body = _avg_body(body, cfg)
    c1_bear = _shift_bool(f.bear, 2)
    c1_bull = _shift_bool(f.bull, 2)
    c2_body = _shift_at(body, 1)
    c2_small = c2_body <= cfg.star_body_max * avg_body
    c1_body = _shift_at(body, 2)
    c1_body_top = _shift_at(body_top, 2)
    c1_body_bot = _shift_at(body_bot, 2)
    c2_body_top = _shift_at(body_top, 1)
    c2_body_bot = _shift_at(body_bot, 1)
    c1_mid = (c1_body_top + c1_body_bot) / 2.0

    if cfg.star_gap_required:
        star_below = c2_body_top < c1_body_bot
        star_above = c2_body_bot > c1_body_top
    else:
        star_below = _shift_at((body_top + body_bot) / 2.0, 1) < c1_mid
        star_above = _shift_at((body_top + body_bot) / 2.0, 1) > c1_mid

    morning_star = (
        valid
        & c1_bear
        & f.bull
        & c2_small
        & star_below
        & (c >= c1_body_bot + cfg.star_retrace_min * c1_body)
    )
    evening_star = (
        valid
        & c1_bull
        & f.bear
        & c2_small
        & star_above
        & (c <= c1_body_top - cfg.star_retrace_min * c1_body)
    )

    three_white_soldiers = _consecutive_soldiers(f, valid, cfg, bullish=True)
    three_black_crows = _consecutive_soldiers(f, valid, cfg, bullish=False)

    data = {
        "doji": doji,
        "hammer": hammer,
        "inverted_hammer": inverted_hammer,
        "shooting_star": shooting_star,
        "marubozu_bullish": marubozu_bullish,
        "marubozu_bearish": marubozu_bearish,
        "bullish_engulfing": bullish_engulfing,
        "bearish_engulfing": bearish_engulfing,
        "bullish_harami": bullish_harami,
        "bearish_harami": bearish_harami,
        "morning_star": morning_star,
        "evening_star": evening_star,
        "three_white_soldiers": three_white_soldiers,
        "three_black_crows": three_black_crows,
    }
    return pd.DataFrame({name: data[name] for name in PATTERN_NAMES}, index=ohlc.index)


def _shift_at(a: np.ndarray, n: int) -> np.ndarray:
    """Shift an array ``n`` bars forward, filling the freshly exposed head with NaN.

    Length-preserving even when ``n`` >= the array length (unlike raw slicing).
    """
    a = np.asarray(a, dtype=float)
    out = np.full(a.shape, np.nan, dtype=float)
    if n == 0:
        return a.copy()
    if 0 < n < a.size:
        out[n:] = a[:-n]
    return out


def _avg_body(body: np.ndarray, cfg: PatternConfig) -> np.ndarray:
    """Rolling average of the body over a short window (star body yardstick)."""
    del cfg  # kept in the signature for symmetry / future windowing options
    s = pd.Series(body).rolling(5, min_periods=1).mean()
    return s.to_numpy(dtype=float)


def _consecutive_soldiers(f: _Features, valid: np.ndarray, cfg: PatternConfig,
                          *, bullish: bool) -> np.ndarray:
    """Three white soldiers / three black crows.

    Three consecutive same-direction candles whose closes march in that
    direction, each with a solid body, a small leading shadow, and an open that
    sits inside the previous real body (the classic continuation rule).
    """
    prev_close = _shift_at(f.close, 1)
    prev2_close = _shift_at(f.close, 2)
    prev_body_bot = _shift_at(f.body_bot, 1)
    prev_body_top = _shift_at(f.body_top, 1)
    if bullish:
        same = f.bull
        prior_dir = _shift_bool(f.bull, 1) & _shift_bool(f.bull, 2)
        leading = f.upper
        march = (f.close > prev_close) & (prev_close > prev2_close)
        inside_prior = (f.open >= prev_body_bot) & (f.open <= prev_body_top)
    else:
        same = f.bear
        prior_dir = _shift_bool(f.bear, 1) & _shift_bool(f.bear, 2)
        leading = f.lower
        march = (f.close < prev_close) & (prev_close < prev2_close)
        inside_prior = (f.open <= prev_body_top) & (f.open >= prev_body_bot)
    body_ok = f.body >= cfg.soldiers_body_min * f.rng
    shadow_ok = leading <= cfg.soldiers_shadow_max * f.rng
    return valid & same & prior_dir & march & inside_prior & body_ok & shadow_ok


def label(ohlc: pd.DataFrame, cfg: PatternConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Attach a boolean column per pattern plus an aggregated ``patterns`` label.

    Returns a copy of ``ohlc`` with, additionally:

    * one boolean column per name in :data:`candles.config.PATTERN_NAMES`;
    * ``n_patterns`` -- how many patterns fired on that bar;
    * ``patterns`` -- comma-separated names (empty string when none).

    Every candle is labelled: bars with no pattern simply have all-False
    booleans and an empty ``patterns`` string.
    """
    hits = detect(ohlc, cfg)
    out = ohlc.copy()
    for name in PATTERN_NAMES:
        out[name] = hits[name].to_numpy()
    names = np.array(PATTERN_NAMES)
    values = hits.to_numpy(dtype=bool)
    joined = np.array([",".join(names[row]) for row in values])
    out["patterns"] = joined
    out["n_patterns"] = values.sum(axis=1)
    return out
