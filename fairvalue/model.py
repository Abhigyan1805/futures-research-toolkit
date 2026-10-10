"""Cost-of-carry pricing of a futures contract.

Idea: buying the asset today and holding it until expiry costs interest on the
money tied up (r) plus storage (u), and earns any yield from holding it
(dividends q for an index, convenience yield c for a commodity).  No-arbitrage
then pins the futures price to the spot price grown at that net carry rate:

    continuous:  F = S * exp((r + u - q - c) * T)
    discrete:    F = S * (1 + r*T) + FV(storage) - FV(income)   (simple interest,
                 here with rates as annual fractions applied as (1 + b)**T)

All rates are annual decimals (0.05 = 5%), T is in years.  Functions are
vectorised: scalars, numpy arrays or pandas Series all work.
"""
import numpy as np


def net_carry(r, storage=0.0, income_yield=0.0, convenience=0.0):
    """Net cost of carry b = r + storage - dividend/income yield - convenience."""
    return r + storage - income_yield - convenience


def fair_value(spot, r, T, storage=0.0, income_yield=0.0, convenience=0.0, method="continuous"):
    """Fair futures price from spot.

    method="continuous": F = S * exp(b*T)
    method="discrete":   F = S * (1 + b)**T  (annual compounding of the net carry)
    """
    b = net_carry(r, storage, income_yield, convenience)
    if method == "continuous":
        return spot * np.exp(b * T)
    if method == "discrete":
        return spot * (1.0 + b) ** T
    raise ValueError(f"unknown method {method!r}")


def basis(spot, futures):
    """Basis = spot - futures (market convention: negative in contango)."""
    return spot - futures


def implied_carry(spot, futures, T, method="continuous"):
    """Net carry rate b that makes the model reproduce the market futures price."""
    ratio = np.asarray(futures, dtype=float) / np.asarray(spot, dtype=float)
    Tn = np.asarray(T, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if method == "continuous":
            out = np.log(ratio) / Tn
        elif method == "discrete":
            out = ratio ** (1.0 / Tn) - 1.0
        else:
            raise ValueError(f"unknown method {method!r}")
    return out


def implied_income_yield(spot, futures, r, T, storage=0.0, convenience=0.0, method="continuous"):
    """Dividend yield q implied by the market price (solve b = r + u - q - c for q)."""
    b = implied_carry(spot, futures, T, method)
    return r + storage - convenience - b


def implied_convenience_yield(spot, futures, r, T, storage=0.0, income_yield=0.0,
                              method="continuous"):
    """Convenience yield c implied by the market price (solve for c)."""
    b = implied_carry(spot, futures, T, method)
    return r + storage - income_yield - b


def gap(model_price, market_price):
    """Model fair value minus market price (points); positive = market looks cheap."""
    return model_price - market_price
