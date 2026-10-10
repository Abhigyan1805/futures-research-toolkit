"""Run the cost-of-carry study on real free data and write results/fairvalue/.

Contracts (all via the shared ``data`` loaders; r = FRED 3-month T-bill, DGS3MO):
  ES  E-mini S&P 500   spot ^GSPC,  dividend yield *assumed* ASSUMED_Q (documented)
  GC  COMEX gold       spot proxy GLD (ounces/share decays by the 0.40% fee)
  CL  NYMEX WTI crude  spot FRED DCOILWTICO (Cushing spot), no yield -> "naive" carry

For each: model fair value vs market futures, the gap (model - market), and the
*implied* yield that would make the model exact (dividend yield for ES, convenience
yield net of storage for GC/CL).  The tenor T of a Yahoo continuous series is
reconstructed from the exchange calendar (``expiry``) - an approximation.
Treasury futures (ZN) are not priced: their cost of carry depends on the
cheapest-to-deliver bond and conversion factor, which free data does not give.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data import load_fred, load_ohlc

from .excel import write_scenario_workbook
from .expiry import days_to_front_expiry
from .model import fair_value, implied_carry
from .scenarios import Inputs, rate_sensitivity

ASSUMED_Q = 0.014          # S&P 500 dividend yield assumption (~1.3-1.6% in 2018-26)
MIN_DAYS = 7               # implied yields are noise when T -> 0, drop the last week
START = "2018-01-01"
GLD_OZ_START, GLD_FEE, GLD_T0 = 0.10, 0.004, 2004.9   # SPDR Gold Shares: 0.1 oz at launch


def _gld_gold_price(gld: pd.Series) -> pd.Series:
    """Gold $/oz implied by GLD: price / ounces-per-share(t); fee erodes ounces/share."""
    yr = gld.index.year + (gld.index.dayofyear - 1) / 365.25
    return gld / (GLD_OZ_START * np.exp(-GLD_FEE * (yr - GLD_T0)))


def _rate(index, offline) -> pd.Series:
    r = load_fred("DGS3MO", START, offline=offline)["DGS3MO"] / 100.0
    return r.reindex(index.union(r.index)).ffill().reindex(index)


def _contract(name, fut, spot, kind, offline, storage=0.0, q=0.0):
    df = pd.concat({"fut": fut, "spot": spot}, axis=1).dropna()
    df = df[(df > 0).all(axis=1)]
    df["r"] = _rate(df.index, offline)
    df["days"] = days_to_front_expiry(df.index, kind)
    df = df.dropna()
    df = df[df.days >= MIN_DAYS].copy()
    T = df.days / 365.0
    df["fair_cont"] = fair_value(df.spot, df.r, T, storage, q)
    df["fair_disc"] = fair_value(df.spot, df.r, T, storage, q, method="discrete")
    df["basis"] = df.spot - df.fut
    df["gap"] = df.fair_cont - df.fut
    df["gap_bps"] = 1e4 * df.gap / df.fut
    b = implied_carry(df.spot, df.fut, T)
    df["implied_yield"] = df.r + storage - q - b     # q (ES) or convenience (GC/CL)
    df.index.name = "date"
    return df


def _stats(name, df, headline):
    g = df.gap_bps
    headline[f"{name}_days"] = int(len(df))
    headline[f"{name}_mean_abs_gap_bps"] = round(float(g.abs().mean()), 1)
    headline[f"{name}_median_abs_gap_bps"] = round(float(g.abs().median()), 1)
    headline[f"{name}_median_gap_bps"] = round(float(g.median()), 1)
    headline[f"{name}_median_implied_yield_pct"] = round(float(df.implied_yield.median() * 100), 2)
    headline[f"{name}_share_backwardation_pct"] = round(float((df.basis > 0).mean() * 100), 1)


def run(out_dir: Path = Path("results/fairvalue"), offline: bool | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h: dict = {"assumed_sp500_dividend_yield_pct": ASSUMED_Q * 100}
    res: dict[str, pd.DataFrame] = {}

    es = load_ohlc("ES=F", START, offline=offline).close
    spx = load_ohlc("^GSPC", START, offline=offline).close
    res["ES"] = _contract("ES", es, spx, "ES", offline, q=ASSUMED_Q)

    try:
        gc = load_ohlc("GC=F", START, offline=offline).close
        gld = load_ohlc("GLD", START, offline=offline).close
        res["GC"] = _contract("GC", gc, _gld_gold_price(gld), "GC", offline)
    except (FileNotFoundError, RuntimeError) as e:
        h["gold_skipped"] = f"no spot proxy ({type(e).__name__})"

    try:
        cl = load_ohlc("CL=F", START, offline=offline).close
        wti = load_fred("DCOILWTICO", START, offline=offline)["DCOILWTICO"]
        res["CL"] = _contract("CL", cl, wti, "CL", offline)
    except (FileNotFoundError, RuntimeError) as e:
        h["crude_skipped"] = f"no spot series ({type(e).__name__})"

    for name, df in res.items():
        df.round(6).to_csv(out_dir / f"{name}_fairvalue.csv")
        _stats(name, df, h)

    # Same ES model with q=0 shows how much of the gap the dividend assumption explains.
    es0 = res["ES"]
    h["ES_mean_abs_gap_bps_if_q_zero"] = round(float(
        (1e4 * (fair_value(es0.spot, es0.r, es0.days / 365) - es0.fut).abs() / es0.fut).mean()), 1)

    # Scenario workbook from the latest ES observation.
    last = es0.iloc[-1]
    base = Inputs(spot=float(last.spot), r=float(last.r), days=float(last.days),
                  income_yield=ASSUMED_Q)
    write_scenario_workbook(out_dir / "fairvalue.xlsx", base, days_list=(base.days, 30, 7, 0),
                            title=f"ES front contract, as of {es0.index[-1].date()}")
    h["ES_base_fair_value"] = round(base.price(), 2)
    h["ES_last_market_price"] = round(float(last.fut), 2)
    h["ES_dFair_per_bp_rate"] = round(rate_sensitivity(base), 3)
    h["scenario_asof"] = str(es0.index[-1].date())

    _plot(res, out_dir / "fairvalue.png")
    (out_dir / "headline.json").write_text(json.dumps(h, indent=2))
    return h


def _plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(len(res), 2, figsize=(11, 3 * len(res)), squeeze=False)
    for i, (n, d) in enumerate(res.items()):
        ax[i, 0].plot(d.index, d.gap_bps, lw=0.6)
        ax[i, 0].set_title(f"{n}: model fair value - market (bps of futures price)")
        ax[i, 1].plot(d.index, d.implied_yield * 100, lw=0.6, color="C1")
        ax[i, 1].set_title(f"{n}: implied {'dividend' if n == 'ES' else 'convenience'} yield (%)")
        if n != "ES":
            ax[i, 1].set_ylim(-60, 60)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    h = run()
    for k, v in h.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
