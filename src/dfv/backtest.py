"""Rolling-origin backtest with an explicit information set per origin."""
from __future__ import annotations

import pandas as pd

from .data import Panel
from .models import BENCHMARKS, CANDIDATES, DIAGNOSTICS

MIN_HISTORY = 5   # features need t-5; direct models need >= 1 base year with t+h <= origin


def history_until(panel: Panel, origin: int) -> pd.DataFrame:
    """The information set at the forecast origin: rows with year <= origin only."""
    return panel.frame[panel.frame["year"] <= origin].copy()


def valid_horizons(panel: Panel, origin: int) -> list[int]:
    """Horizons evaluated at this origin.

    A pair (origin, h) is used only if (a) the target year is observed and
    (b) every model, including direct multi-horizon models, has at least one
    complete training pair (t, t+h) with t-MIN_HISTORY >= first year and
    t+h <= origin.  The same pairs are used for every model, so model
    comparisons are like-for-like.
    """
    first_year = int(panel.frame["year"].min())
    return [
        h
        for h in range(1, panel.max_horizon + 1)
        if origin + h <= panel.last_year and origin - h >= first_year + MIN_HISTORY
    ]


def run_backtest(panel: Panel, models: dict | None = None) -> pd.DataFrame:
    """Forecasts of every model at every valid (origin, horizon), joined to actuals."""
    models = models or {**CANDIDATES, **BENCHMARKS, **DIAGNOSTICS}
    out = []
    for origin in range(panel.first_origin, panel.last_year):
        horizons = valid_horizons(panel, origin)
        if not horizons:
            continue
        hist = history_until(panel, origin)
        for name, fn in models.items():
            future = panel.frame if name in DIAGNOSTICS else None
            f = fn(hist, origin, horizons, panel, future=future)
            f["model"] = name
            out.append(f)
    fc = pd.concat(out, ignore_index=True)
    actual = panel.frame[["market", "year", "demand"]].rename(columns={"year": "target_year", "demand": "actual"})
    fc = fc.merge(actual, on=["market", "target_year"], how="left", validate="many_to_one")
    assert fc["actual"].notna().all(), "every evaluated target must be observed"
    return fc
