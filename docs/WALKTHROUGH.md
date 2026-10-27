# Walkthrough in plain English

For explaining this project in an interview. Numbers are from the one real run in
[`RESULTS.md`](RESULTS.md).

## 1. Cost of carry and basis (`fairvalue/`)

A futures contract is a promise to buy an asset at a fixed price on a future date. You could get
the same exposure by buying the asset now and holding it. Holding costs you **interest** on the money
tied up (r) and **storage** (u), and pays you **income** (dividends q for an index, a convenience
yield c for a commodity). No-arbitrage says the futures price must equal the spot price grown at
that net rate:

`F = S · exp((r + u − q − c) · T)`   (T in years to expiry)

Example: S&P at 7,000, r = 4%, q = 1.4%, T = 0.25 → F = 7000·e^(0.026·0.25) ≈ 7,046. The **basis**
is F − S; it is the carry, and it shrinks to zero as T → 0 (futures converge to spot).

What I did: priced ES (against ^GSPC), gold (against the GLD ETF) and WTI (against FRED spot) every
day, compared with the Yahoo futures price, and reported the gap. ES is close (7.9 bps mean absolute
gap) because index carry is simple and observable. Gold is looser (40.6 bps) because the spot
proxy is an ETF and convenience yield is unobserved. Crude is loosest (78.8 bps mean, 26 bps median):
crude has large, swinging convenience yield and storage costs, plus roll gaps.
I also solve the formula backwards for the **implied yield** that would make the model exact; that
turns a "model error" into a market reading. Scenario grids show fair value under spot, rate and
time shocks (about 0.151 ES points per bp of rate), also exported to Excel with live formulas.

For STIR products the analogue is a rate future priced as 100 − expected
rate; the cost-of-carry idea is the same, but a free-data version for listed STIR contracts
is not possible, so I built the engine on equity, metal and energy futures and said so.

## 2. Calendar spreads and butterflies (`regimes/`)

- **Calendar spread**: far contract minus near contract (here far tenor yield minus near tenor
  yield, in bps). It measures the *slope* of a segment of the curve.
- **Butterfly**: `2·middle − near − far`. It measures *curvature* and is neutral to parallel moves
  and (for a weighted structure) to slope moves, so it isolates a view on the shape.

**Correlation regimes.** Correlation is not a constant: in calm markets different spreads move
fairly independently; in stress they co-move (diversification fails when you need it). I measured
the correlation matrix of daily changes inside calm, normal and stressed regimes. Mean correlation
rises from 0.059 to 0.080 (+36%), and individual pairs flip sign (5y−1y vs a 2·2y−1y−5y fly:
−0.078 → +0.291).

**Regime detection.** Two detectors for cross-checking: a 3-state Gaussian hidden Markov model
(hidden states = calm/normal/stressed; each has its own volatility profile; fitted by
Baum-Welch/EM, written from scratch in NumPy) and a plain quantile threshold on a stress score.
They agree on 76% of days, and the HMM flags 2008, 2020 and 2022. It misses the 2013 taper tantrum,
which was a short sharp episode; I report that instead of tuning it away.

**The proxy.** Listed STIR contract histories are not free, so this uses Treasury and policy
rate series from FRED. Say it first, before they ask.

## 3. Candlestick patterns (`candles/`)

A candle summarises a day: open, high, low, close. **Body** = |close − open|, **range** = high − low,
**shadows** = the thin lines above and below.
- Single: *hammer* (small body, long lower shadow after a decline), *doji* (open ≈ close),
  *marubozu* (all body, no shadows).
- Double: *engulfing* (second body covers the first, opposite colour), *harami* (second body
  inside the first).
- Triple: *morning/evening star* (long candle, small gap candle, long reversal candle),
  *three white soldiers / black crows* (three long same-direction candles).

All thresholds are relative (shadow ≥ N× body, body ≤ x% of range) and live in one config, so the
same rules work on ES at 7,000 and ZN at 110. Every candle gets a label.

**Does it work?** For each pattern and horizon (1, 5, 10 days) I test whether the next move goes the
claimed way more often than the *base rate* (how often the market goes that way anyway), plus a t-test
and a random-entry bootstrap. With 42 tests, about two would look "significant" at 5% by luck, so
p-values are corrected with Benjamini-Hochberg. Result: 4 survive, none in the claimed direction.
Mean hit rate 49.6% vs 50.4% base. Textbook patterns are no better than chance on liquid
futures daily bars, which is plausible: they are public, simple, and traded away.

## Design choices and why

- **Explicit packages + one data layer**: the same cached loader, with fixtures, is used by all modules;
  tests and CI run offline.
- **Honest about proxies**: continuous Yahoo futures, FRED instead of STIR contracts, assumed dividend yield.
- **From-scratch HMM**: nothing hidden behind a library, so every step can be explained.
- **Fixed seeds, one command** (`make e2e`) so the numbers reproduce.

## Likely interview questions

**Why does the futures price differ from spot?** Cost of carry: financing plus storage minus income.
**What is basis and what happens at expiry?** F − S; it converges to zero because the contract becomes the
asset.
**Why is crude's gap bigger?** Convenience yield is unobservable and volatile, storage is costly,
contango/backwardation swings, and the continuous series has roll gaps.
**If model and market differ, is it arbitrage?** Not by itself: transaction costs, financing spread,
unobserved yields and my approximate T explain a lot. The implied-yield view shows what the market
is pricing instead.
**Why butterflies?** They cancel level and slope exposure, leaving a view on curvature; correlations
between structures show whether positions are really diversified.
**Why does correlation rise in stress?** Common factors (policy shocks, liquidity) dominate, so
diversification shrinks exactly when volatility is high.
**What is an HMM in one sentence?** A model where the market sits in an unobserved state, each with its
own behaviour, and the state switches with fixed transition probabilities; EM fits both.
**How do you know the regimes are not overfit?** Second, independent detector (76% agreement),
known-episode checks, and I report the miss (2013).
**Why do most candle patterns fail?** Daily returns are close to a coin flip, patterns are
subjective and widely known, and testing many patterns produces false positives; multiple-testing
correction removes them.
**What is Benjamini-Hochberg?** Controls the expected share of false discoveries among results called
significant, less harsh than Bonferroni.
**What would you do next with real STIR data?** Replace FRED proxies with SR3 contract histories, roll-adjust, and rerun
the same pipeline.
