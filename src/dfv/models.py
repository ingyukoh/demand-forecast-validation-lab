"""Forecasting models evaluated by the backtest.

Every model has the same contract:

    model(history, origin, horizons, panel, future=None) -> DataFrame
        columns: market, origin, horizon, target_year, forecast

``history`` contains ONLY rows with year <= origin.  The backtest enforces
this (see ``backtest.history_until``) and ``tests/test_leakage.py`` checks
that changing any value after the origin leaves every forecast unchanged.

The single exception is ``pooled_drivers_oracle``: it is handed the realised
future driver values through ``future`` on purpose, to measure how much of a
driver's apparent value comes from look-ahead.  It is reported as a
diagnostic ceiling and is never eligible to win a market.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import Panel

PLANNER_GROWTH_CAP = 0.10   # planner proxy never extrapolates more than +/-10%/yr


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _wide(history: pd.DataFrame, col: str) -> pd.DataFrame:
    """market x year matrix of a column (log of demand for 'demand')."""
    w = history.pivot(index="market", columns="year", values=col).sort_index(axis=1)
    return np.log(w) if col == "demand" else w


def _rows(markets, origin, horizons, values) -> pd.DataFrame:
    recs = []
    for h in horizons:
        for m in markets:
            recs.append((m, origin, h, origin + h, float(values(m, h))))
    return pd.DataFrame(recs, columns=["market", "origin", "horizon", "target_year", "forecast"])


# ----------------------------------------------------------------------------
# univariate benchmarks
# ----------------------------------------------------------------------------
def naive(history, origin, horizons, panel, future=None):
    L = _wide(history, "demand")
    last = L[origin]
    return _rows(L.index, origin, horizons, lambda m, h: np.exp(last[m]))


def drift(history, origin, horizons, panel, future=None, window=5):
    """Log-linear drift: average annual log growth over the last ``window`` years."""
    L = _wide(history, "demand")
    g = (L[origin] - L[origin - window]) / window
    return _rows(L.index, origin, horizons, lambda m, h: np.exp(L.at[m, origin] + h * g[m]))


def planner_proxy(history, origin, horizons, panel, future=None):
    """Stand-in for a judgmental plan: compound last year's growth, capped.

    There are no human forecasts in these public panels.  This proxy imitates a
    recency-weighted, sentiment-driven plan ("last year's trend continues") so
    that Forecast Value Added can be demonstrated.  It is an assumption, not a
    measurement of any real planning team.
    """
    L = _wide(history, "demand")
    g = np.clip(np.expm1(L[origin] - L[origin - 1]), -PLANNER_GROWTH_CAP, PLANNER_GROWTH_CAP)
    return _rows(L.index, origin, horizons, lambda m, h: np.exp(L.at[m, origin]) * (1 + g[m]) ** h)


def damped_trend(history, origin, horizons, panel, future=None):
    """Holt's additive damped trend on log demand, fitted per market."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    L = _wide(history, "demand")
    H = max(horizons)
    paths = {}
    for m in L.index:
        y = L.loc[m].dropna().to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ExponentialSmoothing(y, trend="add", damped_trend=True, initialization_method="estimated").fit(
                optimized=True
            )
        paths[m] = fit.forecast(H)
    return _rows(L.index, origin, horizons, lambda m, h: np.exp(paths[m][h - 1]))


# ----------------------------------------------------------------------------
# pooled direct multi-horizon models with structural drivers
# ----------------------------------------------------------------------------
def _feature_block(L, D, t, drivers):
    """Features for base year t using only years <= t."""
    f = pd.DataFrame(index=L.index)
    f["g1"] = L[t] - L[t - 1]
    f["g3"] = (L[t] - L[t - 3]) / 3
    f["g5"] = (L[t] - L[t - 5]) / 5
    for k in drivers:
        f[f"{k}_d1"] = D[k][t] - D[k][t - 1]
        f[f"{k}_d5"] = (D[k][t] - D[k][t - 5]) / 5
        f[f"{k}_lvl"] = D[k][t] - D[k].loc[:, :t].mean(axis=1)   # deviation from own history
    return f


def _training_set(L, D, drivers, origin, h, use_future_drivers):
    years = [c for c in L.columns if c <= origin]
    first = years[0] + 5
    X, y = [], []
    for t in range(first, origin - h + 1):          # t + h <= origin  -> target is observed
        f = _feature_block(L, D, t, drivers)
        if use_future_drivers:                        # realised change t -> t+h, known by origin
            for k in drivers:
                f[f"{k}_fut"] = D[k][t + h] - D[k][t]
        X.append(f)
        y.append(L[t + h] - L[t])
    return pd.concat(X), pd.concat(y)


def _pooled(history, origin, horizons, panel, future=None, *, driver_mode="known", learner="ridge"):
    drivers = panel.drivers
    L = _wide(history, "demand")
    D = {k: _wide(history, k) for k in drivers}
    use_fut = driver_mode in ("forecast", "oracle")
    if driver_mode == "oracle":
        if future is None:
            raise ValueError("oracle mode needs realised future drivers")
        Dfull = {k: _wide(future, k) for k in drivers}

    preds = {}
    for h in horizons:
        X, y = _training_set(L, D, drivers, origin, h, use_fut)
        if learner == "ridge":
            est = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        else:
            est = HistGradientBoostingRegressor(
                max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=20, random_state=0
            )
        est.fit(X.to_numpy(), y.to_numpy())

        f = _feature_block(L, D, origin, drivers)
        if driver_mode == "forecast":                # drivers extrapolated by their own 5y drift
            for k in drivers:
                f[f"{k}_fut"] = h * f[f"{k}_d5"]
        elif driver_mode == "oracle":                # realised values: LOOK-AHEAD, diagnostic only
            for k in drivers:
                f[f"{k}_fut"] = Dfull[k][origin + h] - Dfull[k][origin]
        preds[h] = pd.Series(est.predict(f.to_numpy()), index=f.index)

    return _rows(L.index, origin, horizons, lambda m, h: np.exp(L.at[m, origin] + preds[h][m]))


def pooled_drivers_known(history, origin, horizons, panel, future=None):
    """Ridge on demand momentum + driver changes known at the origin."""
    return _pooled(history, origin, horizons, panel, driver_mode="known")


def pooled_drivers_forecast(history, origin, horizons, panel, future=None):
    """Conditional model: future driver change enters, but is itself forecast (5y drift)."""
    return _pooled(history, origin, horizons, panel, driver_mode="forecast")


def pooled_drivers_oracle(history, origin, horizons, panel, future=None):
    """Same as above with REALISED future drivers. Look-ahead ceiling, never a candidate."""
    return _pooled(history, origin, horizons, panel, future=future, driver_mode="oracle")


def pooled_gbm(history, origin, horizons, panel, future=None):
    """Gradient boosting on the known-at-origin feature set (direct per horizon)."""
    return _pooled(history, origin, horizons, panel, driver_mode="known", learner="gbm")


CANDIDATES = {
    "naive": naive,
    "drift": drift,
    "damped_trend": damped_trend,
    "pooled_drivers_known": pooled_drivers_known,
    "pooled_drivers_forecast": pooled_drivers_forecast,
    "pooled_gbm": pooled_gbm,
}
BENCHMARKS = {"planner_proxy": planner_proxy}
DIAGNOSTICS = {"pooled_drivers_oracle": pooled_drivers_oracle}
ENSEMBLE_MEMBERS = ("drift", "damped_trend", "pooled_drivers_known")
