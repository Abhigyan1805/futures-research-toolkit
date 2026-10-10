# Short-rate spread / butterfly correlation and regime dynamics

PS project 2 ("Correlation study & Regime Dynamics") built as a small, honest
version: quantify and compare correlation structure among short-rate calendar
spreads and butterflies, detect regimes, and show how relationships change
between calm and stressed markets.

Everything here is reproducible from free data. All numbers below come from one
real run (`python -m regimes.run`) on 2026-10-10.

---

## 1. What the study does (plain English)

A rates trader does not only care about the level of the curve; they care about
how the *pieces* move together. We build two families of relative-value proxies
from the US curve:

* **Calendar spreads** - far tenor minus near tenor (e.g. `2y-1y`), the
  steepness of a segment. These are the curve analogues of STIR calendar
  spreads.
* **Butterflies** - curvature, `2 x middle - near - far` (e.g.
  `2*2y - 1y - 5y`), the analogue of a STIR butterfly.

We then measure the correlation structure of these proxies, split the history
into **calm / normal / stressed** regimes, and look at how the correlations
change from one regime to the next.

## 2. Data and the honest proxy caveat

* **Source:** free FRED CSV endpoint (no API key), via the shared loader
  `data.load_fred`: `DGS3MO, DGS1, DGS2, DGS5, DGS10` plus policy references
  `DFF, SOFR`.
* **Caveat, stated plainly:** *individually listed STIR futures contract
  histories are not free*, so **constant-maturity Treasury yields stand in for
  the short-rate curve**. This is a proxy: Treasury yields are cash-bond yields,
  not futures prices, so they carry no roll / financing embedded in the same
  way. For a *relative-value correlation* study the proxy is reasonable because
  we only use differences and co-movement, not absolute pricing. It is not a
  substitute for real futures data in production, and that is said here and in
  the code.
* Constant-maturity series are used for **all** tenors (3m through 10y) so the
  3m point is comparable with 1y+; the discount-basis T-bill series (DTB3/DTB6)
  are deliberately **not** mixed in.
* **Sample:** 1982-01-04 to 2026-10-08, 10,936 usable daily observations
  (3m constant-maturity history starts 1981-09).

## 3. Method

### 3.1 Proxies (basis points)
Calendar spreads: `1y-3m, 2y-1y, 5y-2y, 10y-5y, 5y-1y`.
Butterflies: `2*1y-3m-2y, 2*2y-1y-5y, 2*5y-2y-10y`. Eight proxies total.

### 3.2 Stress features
Two transparent features: the 21-day rolling standard deviation of **daily
changes** in the front-end spread (`2y-1y`) and in the belly butterfly
(`2*2y-1y-5y`), each turned into a z-score using an **expanding** window
(only past data - no look-ahead). Global z-scoring was tried first and behaved
badly: the 1980s high-volatility decade dominated the scale and the 2020/2022
episodes were washed out. Expanding standardisation fixes this and is a
point an interviewer will like.

### 3.3 Two regime detectors
1. **Gaussian hidden Markov model, written from scratch in NumPy**
   (`regimes/hmm.py`): diagonal-covariance 3-state HMM fitted by Baum-Welch EM
   in log space, states decoded with Viterbi. No `hmmlearn` dependency; every
   recursion is visible and auditable.
2. **Percentile-threshold baseline** (`regimes/threshold_regimes`): each day's
   stress score is ranked against its own history (expanding percentile), with
   calm below the 50th percentile and stressed above the 85th.

States are re-ordered so **0 = calm, 2 = stressed** (ranked by the fitted state
mean of the standardised features).

## 4. Headline results (one real run)

| Item | Value |
|---|---|
| Sample | 1982-01-04 -> 2026-10-08 |
| Usable observations | 10,936 |
| Proxies | 8 (5 calendar spreads + 3 butterflies) |
| Regime shares | calm 45.5%, normal 36.7%, stressed 17.8% |
| **Mean pairwise corr of daily changes** | calm **0.059** -> normal 0.068 -> stressed **0.080** (**+36%** in stress) |
| Mean pairwise corr of levels | calm 0.191 -> stressed 0.215 |
| Mean abs correlation shift, calm vs stressed | 0.133 |
| Largest pairwise change-correlation flip | `5y-1y` vs belly fly `2*2y-1y-5y`: **-0.078 -> +0.291** |
| HMM / threshold label agreement | 76.4% |
| HMM log-likelihood | -5194.4 |

### Known stress episodes (do they show up?)

| Episode | Elevated share | Stressed share | Detected |
|---|---|---|---|
| GFC 2008 (2008-09..2009-06) | 100.0% | 83.6% | yes |
| COVID 2020 (2020-02-24..2020-04-30) | 85.4% | 47.9% | yes |
| Hiking cycle 2022 (2022-2023) | 100.0% | 81.8% | yes |
| Taper tantrum 2013 | 53.8% | 0.0% | no (out of scope) |

*Detection* = a majority of the window's days are in a non-calm state. **3 of 3
in-scope short-rate episodes are flagged.**
The 2013 taper tantrum is deliberately marked **out of scope**: it was a
*long-end* (10y) repricing while the policy rate was pinned at zero, so a
short-rate volatility detector correctly does **not** call it stressed. That is
itself the finding - not a failure to hide.

### The economic story
1. **Correlations intensify in stress.** Average pairwise correlation of daily
   spread changes rises from 0.059 (calm) to 0.080 (stressed), +36%. Relative-
   value trades that look diversified in calm markets move together when it
   matters.
2. **The structure reorients, it does not just scale.** Calm-vs-stressed mean
   absolute correlation shift is 0.133, and individual pairs swing hard: the
   `5y-1y` spread vs the belly butterfly flips from **-0.08** (calm) to **+0.29**
   (stressed). Hedges calibrated in calm markets are the wrong hedges in stress.

## 5. Figures and files

`results/regimes/`
* `regime_timeline.png` - proxy series shaded by regime + stress score.
* `rolling_corr.png` - 63-day rolling correlations of spread changes, shaded by regime.
* `corr_changes_heatmaps.png`, `corr_levels_heatmaps.png` - correlation matrix per regime.
* `corr_flip.png` - stressed minus calm correlation heatmap (the "where it moved" map).
* `state_scatter.png` - the two stress features coloured by regime.
* `regimes.xlsx` - Excel summary: regime table, episode coverage, per-regime
  correlation matrices, transition matrices, spread stats, provenance.
* `headline.json`, `regime_summary.csv`, `episode_coverage_*.csv`,
  `transition_*.csv` - the tables above in machine-readable form.

## 6. Design choices and limitations (be ready to defend these)

* **Proxy, not futures** (see section 2). Stated every time it matters.
* **Expanding standardisation**, not global, to avoid look-ahead and
  era-domination.
* **Two detectors** so each cross-checks the other; 76% label agreement is
  healthy given one is parametric and one is a quantile rule.
* **3 states** chosen over 2 for a graded calm/normal/stressed story; the
  calm-vs-stressed comparison uses states 0 and 2. A 2-state version was tried
  and gives essentially the same episode coverage.
* **No predictive claim.** This is a descriptive regime study. Regimes are
  labelled ex-post; turning this into a signal would need a causal / online
  filter and out-of-sample evaluation, which is future work.
* **Gaussian emissions** are a modelling assumption; volatility is persistent
  and fat-tailed, so the HMM is a simplification. It is used for its
  interpretability, not because rates are truly Gaussian.

## 7. Likely interview questions (with answers)

**Q1. Why not use actual STIR futures?**
Contract-level STIR futures history is not free, and the brief required free
data. Treasury constant-maturity yields are the standard free proxy for the
short-rate curve. I only use differences and correlations, so the proxy's level
biases (cash vs futures, discount basis) largely cancel; I state the caveat
rather than hide it.

**Q2. What is a hidden Markov model, and how did you fit it?**
A latent state (calm/normal/stressed) evolves as a Markov chain; each state emits
observations from its own Gaussian. I fit it with Baum-Welch (EM): the
forward-backward algorithm gives the posterior state probabilities and expected
transitions (E-step), then I re-estimate the transition matrix, state means and
variances (M-step), all in log space for numerical stability. States are decoded
with Viterbi. I implemented it from scratch so there is no black box.

**Q3. How do you avoid look-ahead bias?**
The stress features use expanding z-scores and rolling volatilities computed
only from data up to each date. The HMM is fitted on the full sample, which is
an in-sample labelling choice - I flag it explicitly and note that an online
filter is the honest next step if you wanted to trade it.

**Q4. What does the correlation flip actually mean?**
In calm markets curve segments and flies are driven by idiosyncratic flows, so
correlations are low and even negative. Under stress a common factor - policy
expectation repricing and liquidity - dominates, so everything loads on the same
factor, average correlation rises, and previously offsetting positions
converge. That breaks diversification exactly when you need it.

**Q5. Why doesn't the 2013 taper tantrum show up?**
Because it was a long-end event. The Fed's policy rate was pinned at zero and
short-rate volatility stayed low; the repricing happened at the 5y-10y part of
the curve. A short-rate detector should not flag it, and I mark it out of scope
rather than tune the model to force it.

**Q6. Why three states, and how do you know they are real?**
Three states give a usable calm/normal/stressed gradient; the HMM transition
matrix is strongly diagonal (regimes are persistent, not noise), the labels line
up with independently chosen historical episodes, and a completely different
quantile-threshold method agrees on 76% of days. That convergence is the
evidence they are not artefacts.

**Q7. What is the practical takeaway for a rates desk?**
Hedge ratios and diversification assumptions estimated in calm markets are
unstable - correlations can flip sign in stress. Risk should be sized for the
stressed-regime correlation, and spread/butterfly books should be stress-tested
under a high-correlation common-factor regime, not a normal one.

## 8. Reproducing

```bash
make test          # offline unit tests (no network)
python -m regimes.run   # real FRED run -> results/regimes/ (needs network once)
```

Tests cover the spread/butterfly maths, correlation helpers, the HMM (recovers
well-separated synthetic regimes, is seed-reproducible, log-likelihood
increases), the standardisation, the threshold rule, episode coverage and the
transition matrix.
