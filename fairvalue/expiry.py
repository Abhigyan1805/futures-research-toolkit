"""Approximate expiry calendars for the front contract of each free-data series.

Yahoo's ES=F / GC=F / CL=F are *continuous* front-month series: they do not say
which contract was priced.  We reconstruct the likely front contract from the
exchange calendar.  This is an approximation (see docs/fairvalue.md): Yahoo rolls
on its own schedule, so T can be off by up to a few weeks around rolls.
"""
import calendar
from datetime import date, timedelta

import numpy as np
import pandas as pd


def third_friday(y, m):
    d = date(y, m, 1)
    first_fri = d + timedelta(days=(4 - d.weekday()) % 7)
    return first_fri + timedelta(days=14)


def _prev_bdays(d, n):
    while n:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


def es_expiries(y):          # E-mini S&P: Mar/Jun/Sep/Dec, 3rd Friday
    return [third_friday(y, m) for m in (3, 6, 9, 12)]


def gc_expiries(y):          # COMEX gold, active months Feb/Apr/Jun/Aug/Oct/Dec; last
    out = []                 # trading day = 3rd-last business day of the month
    for m in (2, 4, 6, 8, 10, 12):
        d = date(y, m, calendar.monthrange(y, m)[1])
        out.append(_prev_bdays(d + timedelta(days=1), 3))
    return out


def cl_expiries(y):         # NYMEX WTI: monthly, 3 business days before the 25th of
    out = []                # the month *before* delivery (25th rolled back if weekend)
    for m in range(1, 13):
        pm, py = (m - 1, y) if m > 1 else (12, y - 1)
        d25 = date(py, pm, 25)
        while d25.weekday() >= 5:      # 25th not a business day -> use prior business day
            d25 -= timedelta(days=1)
        out.append(_prev_bdays(d25, 3))
    return out


CALENDARS = {"ES": es_expiries, "GC": gc_expiries, "CL": cl_expiries}


def days_to_front_expiry(index, kind, min_days=0):
    """Calendar days from each date to the first expiry >= date + min_days."""
    years = range(index.min().year - 1, index.max().year + 3)
    exps = sorted(e for y in years for e in CALENDARS[kind](y))
    exps = np.array(pd.to_datetime(exps).values, dtype="datetime64[D]")
    dates = np.array(index.values, dtype="datetime64[D]")
    pos = np.searchsorted(exps, dates + np.timedelta64(min_days, "D"))
    return pd.Series((exps[pos] - dates).astype(int), index=index)
