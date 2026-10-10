"""Scenario projection of fair value under rate / spot / time-to-expiry shocks.

A scenario is a (spot shock %, rate shock bp, days-to-expiry) triple applied to a
base set of inputs; the grid is the Cartesian product, so one can read off how
much the fair value moves for each shock and how it converges to spot at expiry.
"""
from dataclasses import dataclass, replace
from itertools import product

from .model import fair_value


@dataclass(frozen=True)
class Inputs:
    spot: float
    r: float
    days: float                 # calendar days to expiry
    storage: float = 0.0
    income_yield: float = 0.0
    convenience: float = 0.0
    method: str = "continuous"

    @property
    def T(self):
        return self.days / 365.0

    def price(self):
        return float(fair_value(self.spot, self.r, self.T, self.storage,
                                self.income_yield, self.convenience, self.method))


def shock(base: Inputs, spot_pct=0.0, rate_bp=0.0, days=None):
    """Return a copy of ``base`` with a relative spot shock, parallel rate shock
    in basis points, and (optionally) a new days-to-expiry."""
    return replace(base, spot=base.spot * (1.0 + spot_pct),
                   r=base.r + rate_bp / 1e4,
                   days=base.days if days is None else days)


def scenario_grid(base: Inputs, spot_pcts=(-0.10, -0.05, 0.0, 0.05, 0.10),
                  rate_bps=(-100, -50, 0, 50, 100), days_list=None):
    """List of dict rows: one per (spot shock, rate shock, days) combination."""
    days_list = tuple(days_list) if days_list is not None else (base.days,)
    base_price = base.price()
    rows = []
    for sp, rb, d in product(spot_pcts, rate_bps, days_list):
        s = shock(base, sp, rb, d)
        p = s.price()
        rows.append({"spot_shock": sp, "rate_shock_bp": rb, "days": d,
                     "spot": s.spot, "r": s.r, "fair_value": p,
                     "change_vs_base": p - base_price})
    return rows


def rate_sensitivity(base: Inputs):
    """dF/dr per +1bp (analytic for continuous: F * T * 1e-4)."""
    return base.price() * base.T * 1e-4
