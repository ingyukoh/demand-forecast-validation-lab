"""Information-set tests: nothing after the origin may change a forecast."""
import numpy as np
import pytest

from dfv.backtest import history_until, valid_horizons
from dfv.data import load_cigar, load_gasoline
from dfv.models import BENCHMARKS, CANDIDATES, pooled_drivers_oracle

PANELS = [load_cigar(), load_gasoline()]


def _scramble_after(panel, origin, seed=1):
    """Copy of the panel frame with every value after the origin replaced by noise."""
    rng = np.random.default_rng(seed)
    f = panel.frame.copy()
    late = f.year > origin
    for c in ["demand", *panel.drivers]:
        f.loc[late, c] = f.loc[late, c] * rng.uniform(0.5, 1.5, late.sum())
    return f


@pytest.mark.parametrize("panel", PANELS, ids=lambda p: p.name)
@pytest.mark.parametrize("name", [*CANDIDATES, *BENCHMARKS])
def test_forecast_ignores_post_origin_data(panel, name):
    origin = panel.selection_end
    hz = valid_horizons(panel, origin)
    fn = {**CANDIDATES, **BENCHMARKS}[name]

    clean = fn(history_until(panel, origin), origin, hz, panel)
    scrambled_panel = type(panel)(**{**panel.__dict__, "frame": _scramble_after(panel, origin)})
    dirty = fn(history_until(scrambled_panel, origin), origin, hz, panel)

    np.testing.assert_allclose(clean.forecast.to_numpy(), dirty.forecast.to_numpy(), rtol=1e-10)


@pytest.mark.parametrize("panel", PANELS, ids=lambda p: p.name)
def test_history_until_has_no_future_rows(panel):
    for origin in range(panel.first_origin, panel.last_year):
        assert history_until(panel, origin).year.max() == origin


@pytest.mark.parametrize("panel", PANELS, ids=lambda p: p.name)
def test_oracle_is_detected_as_lookahead(panel):
    """The test harness must be able to catch a model that peeks: the oracle does."""
    origin = panel.selection_end
    hz = valid_horizons(panel, origin)
    scrambled = _scramble_after(panel, origin)
    a = pooled_drivers_oracle(history_until(panel, origin), origin, hz, panel, future=panel.frame)
    b = pooled_drivers_oracle(history_until(panel, origin), origin, hz, panel, future=scrambled)
    assert not np.allclose(a.forecast, b.forecast)
