"""Configuration for the candlestick pattern engine.

Every threshold that decides whether a candle "counts" is a named field here so
that a reviewer can see, in one place, exactly what the rules are.  The defaults
follow the classic textbook descriptions (Bulkowski, Nison) but are expressed as
relative measures (fractions of the candle's own range or body) rather than
absolute price units, which is what makes them portable across ES, GC, CL, ZN...

The two ideas used throughout:

* *range*  = high - low of the candle.
* *body*   = abs(close - open).

A "small body" is small **relative to its own range** (an indecision candle); a
"long body" is large relative to its own range (a conviction candle).  Shadow
comparisons are made relative to the body (a hammer's lower shadow is *N times
its body*), with a floor on the body so a near-doji does not blow up the ratio.
"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class PatternConfig:
    """All tunable rule parameters.  Frozen so a config is a safe hashable key."""

    # --- generic shape thresholds -------------------------------------------
    doji_body_max: float = 0.10
    """Doji: body <= this fraction of the candle range."""

    small_body_max: float = 0.30
    """Small real body <= this fraction of the candle range (hammer, star)."""

    long_body_min: float = 0.60
    """Long real body >= this fraction of the candle range (trend candles)."""

    body_floor: float = 0.05
    """Body is floored at this fraction of range before shadow ratios are taken."""

    # --- marubozu ------------------------------------------------------------
    marubozu_body_min: float = 0.90
    marubozu_shadow_max: float = 0.05
    """Marubozu: body >= 90% of range and each shadow <= 5% of range."""

    # --- hammer / shooting star ---------------------------------------------
    hammer_shadow_min: float = 2.0
    """Hammer: lower shadow >= this multiple of the body."""

    hammer_opp_shadow_max: float = 0.30
    """Hammer: upper (opposite) shadow <= this fraction of the range."""

    star_shadow_min: float = 2.0
    """Inverted hammer / shooting star: upper shadow >= this multiple of body."""

    star_opp_shadow_max: float = 0.30
    """Inverted hammer / shooting star: lower shadow <= this fraction of range."""

    # --- trend context -------------------------------------------------------
    trend_lookback: int = 5
    """Trend is measured over the previous N candles (excluding the pattern bar)."""

    trend_min_move: float = 0.0
    """Minimum signed prior return over the lookback to call a trend real."""

    require_trend_hammer: bool = True
    require_trend_star: bool = True
    """Require downtrend for hammer/inverted-hammer and uptrend for shooting star."""

    # --- double-candle patterns ---------------------------------------------
    engulf_prior_body_min: float = 0.0
    """Engulfing: prior body must be >= this fraction of its range (0 = off)."""

    harami_body_max: float = 0.60
    """Harami: current body <= this fraction of the prior candle's body."""

    # --- triple-candle patterns ---------------------------------------------
    star_body_max: float = 0.50
    """Morning/evening star: middle body <= this fraction of the recent average."""

    star_gap_required: bool = False
    """If True the star body must be strictly outside the first candle's body."""

    star_retrace_min: float = 0.50
    """Third candle closes past this fraction of the first candle's body."""

    soldiers_body_min: float = 0.50
    """Three soldiers/crows: each body >= this fraction of its range."""

    soldiers_shadow_max: float = 0.30
    """Three soldiers/crows: leading shadow <= this fraction of range."""

    # --- data hygiene --------------------------------------------------------
    min_range_pct: float = 0.0
    """Candles with range <= this fraction of the rolling median are ignored."""

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def replace(self, **kwargs) -> "PatternConfig":
        base = self.as_dict()
        base.update(kwargs)
        return PatternConfig(**base)


DEFAULT_CONFIG = PatternConfig()

#: Patterns produced by the engine, in a stable order.  ``side`` says which
#: direction the pattern is traditionally read as predicting (used later when a
#: hit is scored against the realised forward return).
PATTERN_SPECS: tuple[tuple[str, str], ...] = (
    ("doji", "neutral"),
    ("hammer", "bullish"),
    ("inverted_hammer", "bullish"),
    ("shooting_star", "bearish"),
    ("marubozu_bullish", "bullish"),
    ("marubozu_bearish", "bearish"),
    ("bullish_engulfing", "bullish"),
    ("bearish_engulfing", "bearish"),
    ("bullish_harami", "bullish"),
    ("bearish_harami", "bearish"),
    ("morning_star", "bullish"),
    ("evening_star", "bearish"),
    ("three_white_soldiers", "bullish"),
    ("three_black_crows", "bearish"),
)

PATTERN_NAMES: tuple[str, ...] = tuple(name for name, _ in PATTERN_SPECS)

PATTERN_SIDES: dict[str, str] = {name: side for name, side in PATTERN_SPECS}
