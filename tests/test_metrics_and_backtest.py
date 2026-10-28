import numpy as np
import pytest

from dfv.analysis import split_windows
from dfv.backtest import MIN_HISTORY, run_backtest, valid_horizons
from dfv.data import load_cigar, load_gasoline
from dfv.metrics import add_ensemble, bias, wape
from dfv.models import drift, naive, planner_proxy


def test_wape_and_bias_known_values():
    a, f = [100, 200, 100], [110, 180, 100]
    assert wape(a, f) == pytest.approx(30 / 400)
    assert bias(a, f) == pytest.approx(-10 / 400)


def test_planner_proxy_is_capped():
    panel = load_cigar()
    origin = 1985
    hist = panel.frame[panel.frame.year <= origin]
    f = planner_proxy(hist, origin, [1], panel).set_index("market").forecast
    last = hist[hist.year == origin].set_index("market").demand
    assert ((f / last - 1).abs() <= 0.10 + 1e-12).all()


@pytest.mark.parametrize("loader", [load_cigar, load_gasoline])
def test_valid_horizons_rule(loader):
    p = loader()
    first = p.frame.year.min()
    for origin in range(p.first_origin, p.last_year):
        for h in valid_horizons(p, origin):
            assert origin + h <= p.last_year
            assert origin - h >= first + MIN_HISTORY


def test_backtest_small_run_and_windows():
    p = load_gasoline()
    fc = add_ensemble(run_backtest(p, {"naive": naive, "drift": drift}), ["naive", "drift"])
    assert set(fc.model) == {"naive", "drift", "ensemble"}
    assert (fc.target_year == fc.origin + fc.horizon).all()
    assert np.isfinite(fc.forecast).all() and (fc.forecast > 0).all()
    sel, ev = split_windows(fc, p)
    assert sel.target_year.max() <= p.selection_end      # winners use only observable errors
    assert ev.origin.min() >= p.selection_end
    # naive at horizon h equals the value at the origin
    n = fc[fc.model == "naive"].merge(
        p.frame[["market", "year", "demand"]], left_on=["market", "origin"], right_on=["market", "year"])
    np.testing.assert_allclose(n.forecast, n.demand)


def test_data_shapes():
    c, g = load_cigar(), load_gasoline()
    assert len(c.markets) == 46 and (c.frame.groupby("market").size() == 30).all()
    assert len(g.markets) == 18 and (g.frame.groupby("market").size() == 19).all()
    assert (c.frame.demand > 0).all() and (g.frame.demand > 0).all()
    assert not c.frame.isna().any().any() and not g.frame.isna().any().any()
