# Demand Forecast Validation Lab

A reproducible review of **how a multi-market demand forecast is validated**: rolling-origin backtests out to five years, per-market "winner" rules, Forecast Value Added (FVA) against a planning baseline, and tests of external structural drivers without look-ahead.

It runs on two real public demand panels with structural drivers:

| Panel | Markets | Years | Demand | Drivers |
|---|---|---|---|---|
| Baltagi cigarette demand | 46 US states | 1963–1992 | packs per capita × population | real price, real disposable income, population, neighbouring-state price |
| Baltagi–Griffin gasoline demand | 18 OECD countries | 1960–1978 | gasoline per capita | real income, real gasoline price, cars per capita |

> **Scope and provenance.** Independent portfolio project by Ingyu Koh, built in October 2026 with AI assistance. It is not client work and makes no claim about any company's forecasting system. The "planner" is a **proxy** (see below), because these public panels contain no human forecasts.

## Questions it answers

1. **Backtest design.** Is every forecast made only from data available at its origin? (`tests/test_leakage.py` scrambles every value after the origin and requires identical forecasts. The test also shows that it catches a model that does peek.)
2. **Winner rules.** If the best model is picked per market from past backtest errors, does that choice still win afterwards?
3. **Model versus planner.** Is a reported gain over the planning baseline real once it is broken down by market and horizon and given an uncertainty interval?
4. **Long horizons.** How much independent evidence exists at 4–5 years?
5. **External drivers.** How much of a driver's value survives when the driver itself has to be forecast?

## Method

- **Rolling origins**, horizons of 1–5 years, and direct multi-horizon training only on pairs `(t, t+h)` with `t+h ≤ origin`. All models are scored on the same `(origin, horizon)` pairs.
- **Decision date for winner rules** (1981 for cigarettes, 1971 for gasoline):
  - Selection may use only errors whose target year is on or before the decision date.
  - Evaluation uses forecasts issued from the decision date onward.
  - A *leaky* variant, which uses errors observed after the decision date, is computed only to show how often it changes the choice.
- **Models.** Each is fixed before evaluation; nothing was tuned after looking at the evaluation window.
  - Naive.
  - Log drift (5-year window).
  - Holt damped trend.
  - Pooled ridge regression with drivers in three modes:
    - *known*: driver changes up to the origin only;
    - *forecast*: future driver change, extrapolated by its own drift;
    - *oracle*: realised future drivers. This is look-ahead, reported as a diagnostic ceiling, and never a candidate.
  - Pooled gradient boosting.
  - A fixed equal-weight ensemble.
- **Planner proxy.** Last year's growth compounded and capped at ±10% a year. It imitates a recency-driven plan; it is an assumption, not a measurement.
- **Metrics.** Volume-weighted WAPE and bias, by horizon and by market. FVA is the planner's WAPE minus the model's WAPE, in percentage points, with a 90% interval from 2,000 resamples of markets (seed 0).

## Findings (from `reports/results.md`)

**Cigarettes, 46 states, forecasts issued 1981–1991**

| Rule or model | WAPE when chosen | WAPE afterwards | Bias afterwards |
|---|---|---|---|
| Per-market winner | 3.6% | 8.2% | +7.4% |
| Fixed ensemble | 4.8% | 7.9% | +7.1% |
| Log drift | – | 6.9% | +5.0% |
| Planner proxy | 6.4% | 7.9% | +2.7% |

- **Winner rule.** The winner rule did worse than a fixed ensemble that was never selected. Its excess optimism beyond the ensemble was **+1.5 points**. The chosen model was the best model afterwards in **1 of 46** states, and leaky selection would have changed the choice in **33%** of states.
- **Model versus planner.** The best model, log drift, beats the planner proxy by **1.0 point of WAPE** (90% interval 0.4 to 1.9), and does so in 74% of states. **Its bias is worse**, however: +5.0% against +2.7%, because every statistical model over-forecast the post-1981 decline. An overall accuracy gain and a bias change have to be reported separately.
- **Long horizons.** At 5 years there are 7 origins but only **2 non-overlapping windows**. The FVA interval widens to 0.1–3.2 points.
- **External drivers.** Driver changes known at the origin *hurt* at horizons of 2 years or more: WAPE was 16.3% against 12.4% for drift at 5 years. Forecasting the drivers forward recovers part of the loss (13.4%).

**Gasoline, 18 OECD countries, forecasts issued 1971–1977**

- **Model versus planner.** Every model beats the planner proxy. Naive is best, at **+2.9 points** (interval 0.8 to 5.4).
- **External drivers.** At 5 years, drivers with realised future values reach 9.9% WAPE. Forecast drivers reach only 15.1%. **Most of the drivers' apparent value comes from knowing the future**, which is the main trap when external data is added to a long-horizon forecast.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                      # 24 tests: leakage, metrics, windows, data
python scripts/run_review.py   # about 90 s; writes reports/
```

Outputs:

- `reports/results.md`: all tables.
- `reports/<panel>/*.csv`: the same tables as data.
- `forecasts.csv.gz`: every forecast with its actual.
- `summary.json`.
- `figures/`.

## Limitations

- The panels are small and historical (1960–1992), and their markets are US states and OECD countries, not product lines or SKUs.
- The planner is a proxy, so FVA here shows the *method* and says nothing about real planners.
- Raw optimism is partly mechanical, because the selection window holds more short-horizon errors. That is why the table also reports optimism relative to the fixed ensemble.
- The intervals resample markets but treat years as fixed. With only 2 non-overlapping 5-year windows, the long-horizon conclusions are indicative.

## Layout

```
data/raw/            Cigar.csv, Gasoline.csv (see data/README.md for source and hashes)
src/dfv/data.py      panel loaders: market, year, demand, drivers
src/dfv/models.py    candidates, planner proxy, oracle diagnostic
src/dfv/backtest.py  information set per origin, valid (origin, horizon) pairs
src/dfv/analysis.py  winner rules, FVA with bootstrap, horizon and driver reviews
scripts/run_review.py
tests/
```

Code: MIT licence. Data: see `data/README.md`.
