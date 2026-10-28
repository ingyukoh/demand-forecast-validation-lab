"""Load the two public demand panels into one tidy long format.

Every panel is returned with the columns
    market, year, demand, <driver columns...>
where ``demand`` is a strictly positive volume and drivers are numeric
structural variables observed in the same year.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


@dataclass(frozen=True)
class Panel:
    name: str
    frame: pd.DataFrame          # market, year, demand, drivers...
    drivers: tuple[str, ...]     # driver columns (levels, already in logs)
    first_origin: int            # first forecast origin used in backtests
    selection_end: int           # winner rules may only use errors with target_year <= this
    max_horizon: int = 5

    @property
    def markets(self) -> list[str]:
        return sorted(self.frame["market"].unique())

    @property
    def last_year(self) -> int:
        return int(self.frame["year"].max())


def load_cigar() -> Panel:
    """Baltagi cigarette demand: 46 US states, 1963-1992.

    demand  = packs per capita * population (thousand packs)
    drivers = log real price, log real per-capita disposable income,
              log population, log real minimum price in adjoining states
    """
    d = pd.read_csv(RAW / "Cigar.csv").drop(columns=["Unnamed: 0"], errors="ignore")
    d["year"] = d["year"].astype(int) + 1900
    out = pd.DataFrame(
        {
            "market": "S" + d["state"].astype(int).astype(str).str.zfill(2),
            "year": d["year"],
            "demand": d["sales"] * d["pop"],
            "log_real_price": np.log(d["price"] / d["cpi"] * 100.0),
            "log_real_income": np.log(d["ndi"] / d["cpi"] * 100.0),
            "log_pop": np.log(d["pop"]),
            "log_real_price_neighbour": np.log(d["pimin"] / d["cpi"] * 100.0),
        }
    )
    return Panel(
        name="cigar_us_states",
        frame=out.sort_values(["market", "year"]).reset_index(drop=True),
        drivers=("log_real_price", "log_real_income", "log_pop", "log_real_price_neighbour"),
        first_origin=1972,
        selection_end=1981,
    )


def load_gasoline() -> Panel:
    """Baltagi-Griffin OECD gasoline demand: 18 countries, 1960-1978.

    The file holds logs of gasoline per car and cars per capita, so
    demand = exp(lgaspcar + lcarpcap) is gasoline per capita (index units).
    drivers = log real income per capita, log real gasoline price,
              log cars per capita
    """
    d = pd.read_csv(RAW / "Gasoline.csv").drop(columns=["Unnamed: 0"], errors="ignore")
    out = pd.DataFrame(
        {
            "market": d["country"].str.title(),
            "year": d["year"].astype(int),
            "demand": np.exp(d["lgaspcar"] + d["lcarpcap"]) * 1e4,
            "log_real_income": d["lincomep"],
            "log_real_price": d["lrpmg"],
            "log_cars_per_capita": d["lcarpcap"],
        }
    )
    return Panel(
        name="gasoline_oecd_countries",
        frame=out.sort_values(["market", "year"]).reset_index(drop=True),
        drivers=("log_real_income", "log_real_price", "log_cars_per_capita"),
        first_origin=1966,
        selection_end=1971,
    )


PANELS = {"cigar": load_cigar, "gasoline": load_gasoline}
