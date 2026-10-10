# Cost-of-carry fair-value engine (`fairvalue/`)

Run: `make fairvalue` (writes `results/fairvalue/`: per-contract CSVs, `fairvalue.png`,
`fairvalue.xlsx`, `headline.json`). Tests (offline): `make test` or
`pytest tests/test_fairvalue.py tests/test_fairvalue_run.py`.

## Plain-English explanation
A futures contract is a promise to trade an asset on a future date. You could instead buy the
asset now and hold it, which costs interest on the money (r) and storage (u) and pays you any
income from owning it (dividends q for an index; a "convenience yield" c for a commodity that
is worth having in the warehouse). If the futures price differed from "spot grown at the net
carry rate", someone could lock in a risk-free profit. So

    F = S * exp((r + u - q - c) * T)         (continuous)
    F = S * (1 + r + u - q - c) ** T         (discrete, annual compounding)

The module also runs it backwards: given the market futures price, solve for the yield the
market is implying (dividend yield for the S&P, convenience yield for gold/crude).

## Files
- `model.py` fair value (both variants), basis, implied carry / dividend / convenience yield.
- `scenarios.py` shocks (spot %, rate bp, days to expiry), scenario grid, rate sensitivity.
- `excel.py` workbook: `Inputs` sheet + `Scenarios` sheet where fair values are live Excel
  formulas (blue = editable input). Change an input and the grid recalculates.
- `expiry.py` exchange-calendar rules to reconstruct time to expiry for ES, GC, CL.
- `run.py` the real-data study via the shared `data` loaders.

## Data and assumptions
- r: FRED `DGS3MO` (3-month Treasury, constant maturity), forward-filled to futures dates.
- ES=F vs `^GSPC`; S&P dividend yield **assumed 1.4%** (flat, my assumption, not fetched).
- GC=F vs spot **proxy**: GLD price divided by ounces/share (0.1 oz in 2004 shrinking 0.40%/yr
  by the fund fee). Yahoo has no free spot gold; this proxy carries its own error (premium /
  discount to NAV, fee approximation).
- CL=F vs FRED `DCOILWTICO` (Cushing WTI spot). Storage and convenience set to 0 ("naive").
- Time to expiry is reconstructed from the exchange calendar (ES 3rd Friday of Mar/Jun/Sep/Dec;
  GC last-3rd-business-day of Feb/Apr/Jun/Aug/Oct/Dec; CL 3 business days before the 25th of
  the prior month). Last 7 days before expiry are dropped (implied yields blow up as T->0).
- Sample: 2018-01-01 to 2026-10-09. Treasury futures (ZN) are **not** priced: their carry
  depends on the cheapest-to-deliver bond and conversion factor, not available free.

## Headline results (real run, 2026-10-10)
| contract | days | mean abs gap (bps of F) | median abs gap | median implied yield |
|---|---|---|---|---|
| ES (q=1.4% assumed) | 2033 | 7.9 | 5.4 | implied dividend yield 1.28% |
| GC (no convenience) | 1954 | 40.6 | 34.0 | implied convenience-less-storage 1.10% |
| CL (naive carry) | 1687 | 78.8 | 26.2 | implied convenience-less-storage 3.96% |

- ES: with q=0 the mean abs gap is 17.9 bps; with q=1.4% it is 7.9 bps, so the dividend term
  explains most of the carry. The implied 1.28% is close to the S&P's actual yield.
- CL: spot is above the front future on 61.3% of days (backwardation). Naive carry says the
  future should be above spot by the interest rate; the market says otherwise, so the implied
  convenience yield is ~4% median (large and unstable; the mean abs gap is 3x the median,
  by year the mean abs gap ranges from 12 bps in 2021 to 217 bps in 2026, with 2020 at 92 and 2022 at 77). That is the point: for commodities the convenience yield is the
  main unknown, not a rounding error.
- GC: gap is mostly the proxy error plus a persistent ~1% implied number that should be near 0.
- Scenario (ES front contract, as of 2026-10-09): model fair value 7854.05 vs last market
  7859.75; +1 bp in r moves fair value by about 0.151 index points.

## Roll / continuous-contract limitation
Yahoo's `ES=F`, `GC=F`, `CL=F` are stitched front-month series, not a single contract, and they
are not back-adjusted. The price jumps on roll dates by the calendar spread, and Yahoo does not
state which contract month it is quoting or when it rolls. I reconstruct T from the exchange
calendar, so T (and therefore the gap) can be wrong near rolls. This is why the study reports
medians next to means and drops the last 7 days. A proper study needs individual contract
histories, which are not free.

## Interview questions
1. **Why does the futures price exceed spot?** Holding the asset costs financing; futures buyers
   avoid that outlay, so the price carries r*T (net of dividends/convenience). At expiry F=S.
2. **Continuous vs discrete?** Same idea; exp(bT) is the limit of compounding more often. The
   difference is tiny at short T (the tests check both), exp is easier to invert for implied yields.
3. **Why does crude break naive carry?** Storage is costly and physical barrels are valuable
   when scarce. Convenience yield can exceed r+u, so futures sit below spot (backwardation);
   here 61% of days. The model then needs an implied convenience yield, not a constant.
4. **How do you treat the dividend for the S&P?** It is an assumption (1.4%). Backing it out
   from the market gave 1.28%, a sanity check; without it the mean error roughly doubles.
5. **What are the main sources of error?** The assumed yield, the spot proxy (GLD), T from the
   continuous series around rolls, and using a 3M rate for all tenors.
6. **What does +1 bp of rates do?** dF/dr = F*T, i.e. F*T*0.0001; verified against a bump in a test.
