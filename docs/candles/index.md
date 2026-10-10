# Candlestick pattern engine (`candles/`)

A reproducible pipeline that **labels every daily candle** of several liquid
futures with 14 classic candlestick patterns and then asks the only question
that matters for trading: *do those patterns actually predict the next 1, 5 or
10-day return, better than chance?* The answer, on 17 years of real data, is
mostly **no** -- and the report says so with the numbers.

This is PS-II project 3 (Robust Candlestick Pattern Recognition) turned into a
small, honest research tool.

## What it produces

`python -m candles.run` loads real daily OHLC, labels every candle, tests every
pattern, and writes:

| file | content |
| --- | --- |
| `results/candles/labelled_<TICKER>.csv` | every bar with its OHLC, the 14 boolean pattern columns, a `patterns` string and forward returns |
| `results/candles/summary.csv` | one row per pattern x horizon: counts, hit rate, base rate, edge, p-values, BH flag |
| `results/candles/summary` sheet in `candles.xlsx` | the same table plus per-instrument counts, for Excel review |
| `results/candles/hit_rates.png` | hit rate minus base rate by pattern/horizon, BH-significant bars outlined |
| `results/candles/headline.json` | the headline numbers consumed by `python -m data.gather` |

## Rules (explicit and parameterised)

Every threshold is a named field in `candles/config.py::PatternConfig`; the
defaults below are the textbook (Nison/Bulkowski) shapes. `body = |close-open|`,
`range = high-low`, and shadow comparisons use a body floored at 5% of the range
so a near-doji cannot blow up a ratio.

| pattern | rule | trend context |
| --- | --- | --- |
| **doji** | body <= 10% of range (indecision) | none |
| **hammer** | small body, lower shadow >= 2x body, upper shadow <= 30% of range | prior 5-bar move down |
| **inverted hammer** | mirror of hammer (long upper shadow) | prior 5-bar move down |
| **shooting star** | same shape as inverted hammer | prior 5-bar move up |
| **marubozu (bull/bear)** | body >= 90% of range, both shadows <= 5% of range | none |
| **bullish/bearish engulfing** | opposite-colour prior bar; current real body fully engulfs the prior real body | none |
| **bullish/bearish harami** | opposite-colour prior bar; current body sits inside prior body and is <= 60% of it | none |
| **morning/evening star** | long trend bar, small middle star on the far side, third bar closes back past the first body's midpoint | none |
| **three white soldiers / black crows** | three marching same-colour closes, solid bodies, small leading shadows, each open inside the previous real body | its own 3-bar run |

`require_trend_hammer` / `require_trend_star` switch the trend filter on/off;
`trend_lookback`, `trend_min_move`, `doji_body_max`, `hammer_shadow_min`, etc.
change the tolerance. Nothing is hard-coded to a price level.

## Design choices

* **Relative, not absolute, thresholds.** The same rules work on ES (~5000),
  GC (~2000) and ZN (~110) because every test is a ratio.
* **Vectorised and look-ahead-free.** Detectors are boolean numpy operations on
  shifted columns; a bar may only see itself and the two bars before it. Forward
  returns are a separate, clearly later step, so a pattern can never "know" the
  future. `test_no_lookahead_trend_uses_past_only` guards this.
* **Explicit trend context** separates visually identical reversal candles
  (hammer vs hanging man, inverted hammer vs shooting star).
* **No silent cherry-picking.** Every one of the 42 pattern x horizon tests is
  reported, with Benjamini-Hochberg correction, and the count that survives is
  in the headline.

## Data (all free, no proxies needed)

Daily OHLC via the shared loaders (`data.load_ohlc`, Yahoo Finance, unadjusted),
2009-01-01 to today for `ES=F, NQ=F, GC=F, CL=F, ZN=F`. Continuous futures are
Yahoo's front-month series and are **not** roll-adjusted, so a handful of bars
sit on roll dates; that is a mild source of noise, noted rather than hidden.
Tests and CI run fully offline against the committed fixtures.

## Headline numbers (real run)

From `python -m candles.run` over **22,351 daily bars** across 5 futures,
**2009-01-02 to 2026-10-09**, **11,802 pattern events**, **42 tests**:

* **4** pattern x horizon tests survive Benjamini-Hochberg at 5%.
  They are `doji` (5d and 10d, tested via mean return as a neutral pattern) and
  `shooting_star` (1d) and `morning_star` (1d).
* **0 of them point in the textbook direction.** `shooting_star` (a bearish
  candle) is followed by *fewer* down moves than the base rate -- edge **-6.1
  percentage points**. `morning_star` (bullish) is followed by fewer up moves --
  edge **-5.2 pp**. So the significant result is "these patterns carry
  information", and the information runs *opposite* to the folk interpretation.
* Mean hit rate across all directional cells is **49.6%** versus a mean base
  rate of **50.4%**. On this sample the patterns are, if anything, slightly
  *worse* than just taking the unconditional direction.
* Best raw edge in the claimed direction: `three_white_soldiers` at 10 days
  (hit rate 61.9% vs base 57.1%, one-sided p = 0.012) -- but it does **not**
  survive BH correction across 42 tests, so it is not reported as an edge.

The honest one-line conclusion: **classic candlestick patterns do not deliver a
reliable directional edge on liquid futures; the ones that are statistically
distinguishable tend to be contrarian.** That is a finding, not a failure.

## Statistical method

For each pattern and horizon `h` in {1, 5, 10}:

1. **Directional binomial test.** A "hit" is a forward move in the pattern's
   claimed direction. Under the null the hit probability equals the
   *unconditional* base rate of that direction, so the test controls for the
   market's drift. Two-sided p is the primary test; a one-sided directional p is
   also reported (`binom_p_dir`) to separate "deviates from base" from "leans the
   claimed way".
2. **One-sample t-test** of the pattern's mean forward return against zero.
3. **Random-entry bootstrap** (`n_boot=2000`, seed 0): draw the same number of
   entries at random dates and compare mean returns -- a matched random baseline.

Multiple testing: `primary_p` (the two-sided binomial, or the t-test for the
neutral doji) is passed through **Benjamini-Hochberg**; `bh_significant` marks
the cells that survive at alpha = 0.05.

## Limitations

* Yahoo front-month futures are not back-adjusted, so a few bars are roll bars.
* Forward returns at h = 5, 10 are **overlapping** across nearby signals; this
  inflates effective sample size slightly. Counts and p-values are reported
  as-is rather than adjusted, and the bootstrap is the more conservative check.
* Only fixed default parameters are tested; no parameter optimisation, no
  out-of-sample split. This is a labelled-data and significance study, not a
  strategy backtest (no costs, fills or position sizing).

## How to run

```bash
make candles            # online, writes results/candles/
python -m candles.run   # same
make test               # offline unit + pipeline tests
FRT_OFFLINE=1 python -m candles.run   # offline smoke run on fixtures
```

## Likely interview questions

**Q: Why use the base rate rather than 50% in the binomial test?**
A: Because the market drifts. If up-moves happen 53% of the time, a bullish
pattern needs to beat 53%, not 50%, to add information. Using the unconditional
base rate of the same direction removes that free gift.

**Q: Two of your significant patterns are significant in the wrong direction --
isn't that contradictory?**
A: No. The primary test is two-sided: it asks "does the pattern carry
information about the next move?", and `shooting_star`/`morning_star` do, but the
move leans opposite to the classic reading -- consistent with short-horizon mean
reversion after a reversal-looking bar. I report the one-sided directional
p-value too, and **zero** cells are significant in the claimed direction, which
is the honest edge claim.

**Q: Wouldn't testing 42 hypotheses at 5% produce false positives anyway?**
A: Exactly, which is why I apply Benjamini-Hochberg and report the false
discovery rate directly. Only 4 of 42 survive, and I do not claim any of them as
a tradeable edge.

**Q: How do you avoid look-ahead bias?**
A: Detection only ever looks at the current bar and up to two earlier bars
(`shift` operations). Forward returns are a separate step that starts at the
close of the signal bar. A unit test asserts that appending future bars does not
change an earlier bar's label.

**Q: Why relative thresholds instead of fixed point/percent moves?**
A: A hammer on ZN moves by basis points; the same fixed threshold on CL would
never fire. Expressing shadows as multiples of the body and the body as a
fraction of the range makes one rule set valid across instruments.

**Q: What would you do next to make this tradeable?**
A: Add transaction costs, remove overlapping windows, split the sample into
in-sample/out-of-sample, condition patterns on a volatility regime, and require
the edge to survive a walk-forward test -- none of which the current study
claims.

## Files

```
candles/config.py        all rule parameters + the pattern registry
candles/patterns.py      vectorised detectors and the labeller
candles/significance.py  binomial/t/bootstrap tests + BH correction
candles/pipeline.py      label a universe and pool the event study
candles/run.py           entry point (python -m candles.run)
candles/tests/           offline unit and pipeline tests
```
