"""Volume-weighted forecast metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def wape(actual, forecast) -> float:
    """Weighted absolute percentage error: sum|F-A| / sum A."""
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    return float(np.abs(f - a).sum() / a.sum())


def bias(actual, forecast) -> float:
    """Signed volume bias: sum(F-A) / sum A  (>0 means over-forecast)."""
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    return float((f - a).sum() / a.sum())


def summarise(fc: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = fc.groupby(by, observed=True)
    return pd.DataFrame(
        {
            "wape": g.apply(lambda d: wape(d.actual, d.forecast)),
            "bias": g.apply(lambda d: bias(d.actual, d.forecast)),
            "n": g.size(),
        }
    ).reset_index()


def add_ensemble(fc: pd.DataFrame, members, name="ensemble") -> pd.DataFrame:
    """Geometric-mean combination of member forecasts (equal weights, fixed in advance)."""
    keys = ["market", "origin", "horizon", "target_year"]
    sub = fc[fc.model.isin(members)]
    ens = sub.assign(lf=np.log(sub.forecast)).groupby(keys, as_index=False).agg(lf=("lf", "mean"), actual=("actual", "first"))
    ens["forecast"] = np.exp(ens.pop("lf"))
    ens["model"] = name
    return pd.concat([fc, ens[fc.columns]], ignore_index=True)
