"""Review analyses: winner rules, Forecast Value Added, drivers, long horizons."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Panel
from .metrics import bias, summarise, wape
from .models import CANDIDATES

RNG_SEED = 0
N_BOOT = 2000


def split_windows(fc: pd.DataFrame, panel: Panel):
    """Selection = errors already observable at the decision date.
    Evaluation = forecasts issued on or after the decision date."""
    sel = fc[fc.target_year <= panel.selection_end]
    ev = fc[fc.origin >= panel.selection_end]
    return sel, ev


# ----------------------------------------------------------------------------
# winner rules
# ----------------------------------------------------------------------------
def per_market_winners(window: pd.DataFrame, candidates) -> pd.Series:
    s = summarise(window[window.model.isin(candidates)], ["market", "model"])
    return s.loc[s.groupby("market").wape.idxmin()].set_index("market").model


def apply_winners(fc: pd.DataFrame, winners: pd.Series, name: str) -> pd.DataFrame:
    keyed = fc.merge(winners.rename("winner"), left_on="market", right_index=True)
    out = keyed[keyed.model == keyed.winner].drop(columns="winner").copy()
    out["model"] = name
    return out


def winner_rule_review(fc: pd.DataFrame, panel: Panel) -> dict:
    cands = list(CANDIDATES) + ["ensemble"]
    sel, ev = split_windows(fc, panel)

    proper = per_market_winners(sel, cands)
    # LEAKY variant: uses every error from origins before the decision date,
    # including targets that are only observed AFTER it.
    leaky_window = fc[fc.origin < panel.selection_end]
    leaky = per_market_winners(leaky_window, cands)

    s_sel = summarise(sel[sel.model.isin(cands)], ["model"]).set_index("model")
    global_best = s_sel.wape.idxmin()

    rows = []
    for label, winners, window in (("per-market winner (proper)", proper, sel),
                                   ("per-market winner (leaky selection)", leaky, leaky_window)):
        chosen_sel = apply_winners(window, winners, label)
        chosen_ev = apply_winners(ev, winners, label)
        rows.append(
            dict(rule=label,
                 selection_wape=wape(chosen_sel.actual, chosen_sel.forecast),
                 evaluation_wape=wape(chosen_ev.actual, chosen_ev.forecast),
                 evaluation_bias=bias(chosen_ev.actual, chosen_ev.forecast))
        )
    for label, model in ((f"single global model ({global_best})", global_best), ("fixed ensemble", "ensemble"),
                         ("planner proxy", "planner_proxy")):
        a, b = sel[sel.model == model], ev[ev.model == model]
        rows.append(dict(rule=label, selection_wape=wape(a.actual, a.forecast),
                         evaluation_wape=wape(b.actual, b.forecast), evaluation_bias=bias(b.actual, b.forecast)))
    table = pd.DataFrame(rows)
    # Selection and evaluation windows have different horizon mixes, so raw
    # optimism is partly mechanical.  Compare it with the fixed ensemble, which
    # was never selected on data, to isolate the selection effect.
    table["optimism"] = table.evaluation_wape - table.selection_wape
    base = table.loc[table.rule == "fixed ensemble", "optimism"].iloc[0]
    table["excess_optimism_vs_fixed_ensemble"] = table.optimism - base

    expost = per_market_winners(ev, cands)
    return dict(
        table=table,
        winner_counts=proper.value_counts().rename("markets"),
        share_winner_matches_expost_best=float((proper == expost.reindex(proper.index)).mean()),
        share_leaky_differs=float((proper != leaky.reindex(proper.index)).mean()),
        global_best=global_best,
        proper_winners=proper,
    )


# ----------------------------------------------------------------------------
# Forecast Value Added vs the planner proxy, with market-bootstrap intervals
# ----------------------------------------------------------------------------
def _market_boot(ev: pd.DataFrame, model: str, base: str, rng) -> np.ndarray:
    a = ev[ev.model == model].groupby("market")
    b = ev[ev.model == base].groupby("market")
    abs_m = a.apply(lambda d: np.abs(d.forecast - d.actual).sum())
    abs_b = b.apply(lambda d: np.abs(d.forecast - d.actual).sum())
    vol = a.actual.sum()
    mk = abs_m.index.to_numpy()
    draws = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.choice(mk, size=len(mk), replace=True)
        draws[i] = (abs_b[idx].sum() - abs_m[idx].sum()) / vol[idx].sum()
    return draws


def fva_table(ev: pd.DataFrame, models, base="planner_proxy") -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    b = ev[ev.model == base]
    w_base, b_base = wape(b.actual, b.forecast), bias(b.actual, b.forecast)
    per_mkt_base = summarise(b, ["market"]).set_index("market").wape
    rows = []
    for m in models:
        d = ev[ev.model == m]
        per_mkt = summarise(d, ["market"]).set_index("market").wape
        draws = _market_boot(ev, m, base, rng)
        rows.append(dict(model=m, wape=wape(d.actual, d.forecast), planner_wape=w_base,
                         fva_pts=100 * (w_base - wape(d.actual, d.forecast)),
                         fva_ci_low=100 * np.quantile(draws, 0.05), fva_ci_high=100 * np.quantile(draws, 0.95),
                         share_markets_better=float((per_mkt < per_mkt_base.reindex(per_mkt.index)).mean()),
                         bias=bias(d.actual, d.forecast), planner_bias=b_base))
    return pd.DataFrame(rows).sort_values("fva_pts", ascending=False)


# ----------------------------------------------------------------------------
# drivers and horizons
# ----------------------------------------------------------------------------
def driver_review(ev: pd.DataFrame) -> pd.DataFrame:
    keep = ["drift", "pooled_drivers_known", "pooled_drivers_forecast", "pooled_drivers_oracle"]
    s = summarise(ev[ev.model.isin(keep)], ["model", "horizon"])
    return s.pivot(index="horizon", columns="model", values="wape")[keep]


def horizon_table(ev: pd.DataFrame, models) -> pd.DataFrame:
    s = summarise(ev[ev.model.isin(models)], ["model", "horizon"])
    return s.pivot(index="horizon", columns="model", values="wape")[list(models)]


def horizon_fva(ev: pd.DataFrame, model: str, base="planner_proxy") -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for h, d in ev.groupby("horizon"):
        draws = _market_boot(d, model, base, rng)
        a, b = d[d.model == model], d[d.model == base]
        rows.append(dict(horizon=h, origins=a.origin.nunique(),
                         non_overlapping_windows=int(np.ceil(a.origin.nunique() / h)),
                         fva_pts=100 * (wape(b.actual, b.forecast) - wape(a.actual, a.forecast)),
                         ci_low=100 * np.quantile(draws, 0.05), ci_high=100 * np.quantile(draws, 0.95)))
    return pd.DataFrame(rows)
