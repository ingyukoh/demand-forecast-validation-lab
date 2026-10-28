"""Run the full review on both panels and write reports/.

    python scripts/run_review.py            # both panels
    python scripts/run_review.py cigar      # one panel
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dfv.analysis import (  # noqa: E402
    driver_review,
    fva_table,
    horizon_fva,
    horizon_table,
    split_windows,
    winner_rule_review,
)
from dfv.backtest import run_backtest  # noqa: E402
from dfv.data import PANELS  # noqa: E402
from dfv.metrics import add_ensemble  # noqa: E402
from dfv.models import CANDIDATES, ENSEMBLE_MEMBERS  # noqa: E402

REPORTS = ROOT / "reports"


def pct(x):
    return f"{100 * x:.1f}%"


def md(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: floatfmt.format(v))
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "---|" * len(d.columns)
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def review(key: str) -> dict:
    panel = PANELS[key]()
    out = REPORTS / panel.name
    (out / "figures").mkdir(parents=True, exist_ok=True)

    fc = add_ensemble(run_backtest(panel), ENSEMBLE_MEMBERS)
    fc.sort_values(["model", "origin", "horizon", "market"]).to_csv(out / "forecasts.csv.gz", index=False)
    sel, ev = split_windows(fc, panel)

    models = [*CANDIDATES, "ensemble"]
    wr = winner_rule_review(fc, panel)
    from dfv.analysis import apply_winners
    ev_plus = pd.concat([ev, apply_winners(ev, wr["proper_winners"], "winner_rule")], ignore_index=True)

    fva = fva_table(ev_plus, [*models, "winner_rule"])
    hz = horizon_table(ev, ["planner_proxy", *models])
    drv = driver_review(ev)
    best = fva.iloc[0].model
    hfva = horizon_fva(ev_plus, best)

    for name, df in (("winner_rules", wr["table"]), ("fva", fva), ("wape_by_horizon", hz.reset_index()),
                     ("drivers_by_horizon", drv.reset_index()), ("fva_by_horizon", hfva)):
        df.to_csv(out / f"{name}.csv", index=False)

    # figures ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in ["planner_proxy", "naive", "drift", "damped_trend", "pooled_drivers_known", "ensemble"]:
        ax.plot(hz.index, 100 * hz[m], marker="o", label=m)
    ax.set_xlabel("horizon (years)"); ax.set_ylabel("WAPE (%)"); ax.set_xticks(hz.index)
    ax.set_title(f"{panel.name}: error grows with horizon (evaluation window)")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "figures" / "wape_by_horizon.png", dpi=130); plt.close(fig)

    t = wr["table"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = range(len(t))
    ax.bar([i - .2 for i in x], 100 * t.selection_wape, .4, label="selection window")
    ax.bar([i + .2 for i in x], 100 * t.evaluation_wape, .4, label="later evaluation")
    ax.set_xticks(list(x)); ax.set_xticklabels(t.rule, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("WAPE (%)"); ax.set_title("Winner rules: score when chosen vs score afterwards")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=.3, axis="y")
    ax.set_ylim(0, 100 * t[["selection_wape", "evaluation_wape"]].to_numpy().max() * 1.25)
    fig.tight_layout(); fig.savefig(out / "figures" / "winner_rules.png", dpi=130); plt.close(fig)

    summary = dict(
        panel=panel.name,
        markets=len(panel.markets),
        years=[int(panel.frame.year.min()), panel.last_year],
        origins=[int(fc.origin.min()), int(fc.origin.max())],
        decision_date=panel.selection_end,
        forecasts=int(len(fc)),
        evaluation_rows_per_model=int(len(ev[ev.model == "naive"])),
        best_model_by_fva=best,
        winner_counts=wr["winner_counts"].to_dict(),
        share_winner_matches_expost_best=wr["share_winner_matches_expost_best"],
        share_markets_where_leaky_selection_changes_winner=wr["share_leaky_differs"],
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # markdown --------------------------------------------------------------
    lines = [
        f"## {panel.name}",
        "",
        f"{len(panel.markets)} markets, {panel.frame.year.min()}-{panel.last_year}; rolling origins "
        f"{fc.origin.min()}-{fc.origin.max()}, horizons 1-{panel.max_horizon} years. "
        f"Decision date for winner rules: **{panel.selection_end}** (only errors whose target year is "
        f"<= {panel.selection_end} may be used to choose; evaluation uses forecasts issued from "
        f"{panel.selection_end} onward).",
        "",
        "### 1. Winner rules",
        "",
        md(t),
        "",
        f"- Per-market winners chosen: {', '.join(f'{k} ({v})' for k, v in wr['winner_counts'].items())}.",
        f"- The proper winner equals the ex-post best model in {pct(wr['share_winner_matches_expost_best'])} of markets.",
        f"- Leaky selection changes the chosen model in {pct(wr['share_leaky_differs'])} of markets.",
        "",
        "### 2. Forecast Value Added vs planner proxy (evaluation window)",
        "",
        "FVA = planner WAPE minus model WAPE, in percentage points; 90% interval from resampling markets.",
        "",
        md(fva[["model", "wape", "fva_pts", "fva_ci_low", "fva_ci_high", "share_markets_better", "bias"]]),
        "",
        "### 3. WAPE by horizon (evaluation window)",
        "",
        md(hz.reset_index()),
        "",
        f"### 4. Long-horizon evidence for `{best}` vs planner proxy",
        "",
        md(hfva),
        "",
        "### 5. External drivers: known vs forecast vs oracle (WAPE by horizon)",
        "",
        md(drv.reset_index()),
        "",
        "`pooled_drivers_oracle` uses realised future drivers (look-ahead). The gap between it and "
        "`pooled_drivers_forecast` is the part of a driver's apparent value that is not available at "
        "forecast time.",
        "",
        f"![wape by horizon]({panel.name}/figures/wape_by_horizon.png)",
        f"![winner rules]({panel.name}/figures/winner_rules.png)",
        "",
    ]
    return dict(summary=summary, markdown="\n".join(lines))


def main(keys):
    REPORTS.mkdir(exist_ok=True)
    parts = [review(k) for k in keys]
    header = ("# Results\n\nGenerated by `python scripts/run_review.py`. All numbers are reproducible "
              "from the committed data with the pinned requirements.\n")
    (REPORTS / "results.md").write_text(header + "\n" + "\n".join(p["markdown"] for p in parts))
    for p in parts:
        print(json.dumps(p["summary"], indent=2))


if __name__ == "__main__":
    main(sys.argv[1:] or list(PANELS))
