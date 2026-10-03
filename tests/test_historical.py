import numpy as np
import pandas as pd
import pytest

from quant_pod.marketmaking.historical import run_market_maker_on_bars
from quant_pod.marketmaking.markout import compute_markouts
from quant_pod.marketmaking.simulator import summarize


def _bars(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame({"close": closes}, index=idx)


def _synthetic_closes(n: int, sigma: float, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    return list(100.0 + np.cumsum(rng.normal(0, sigma, size=n)))


def test_output_lengths_match_input_bars():
    bars = _bars(_synthetic_closes(50, 1.0, seed=1))
    result = run_market_maker_on_bars(bars, gamma=0.1, kappa=1.5, seed=7)
    lengths = {len(result.mid), len(result.bid), len(result.ask), len(result.inventory), len(result.cash), len(result.equity)}
    assert lengths == {50}


def test_deterministic_given_seed():
    bars = _bars(_synthetic_closes(100, 1.0, seed=1))
    a = run_market_maker_on_bars(bars, gamma=0.1, kappa=1.5, arrival_rate=2.0, seed=7)
    b = run_market_maker_on_bars(bars, gamma=0.1, kappa=1.5, arrival_rate=2.0, seed=7)
    assert np.array_equal(a.inventory, b.inventory)
    assert np.array_equal(a.equity, b.equity)


def test_inventory_starts_pristine_and_equity_is_self_consistent():
    bars = _bars(_synthetic_closes(200, 1.0, seed=2))
    result = run_market_maker_on_bars(bars, gamma=0.1, kappa=1.5, arrival_rate=2.0, seed=7)
    assert result.inventory[0] == 0.0
    assert np.allclose(result.equity, result.cash + result.inventory * result.mid)


def test_mid_tracks_the_input_closes_exactly():
    closes = _synthetic_closes(30, 1.0, seed=3)
    bars = _bars(closes)
    result = run_market_maker_on_bars(bars, gamma=0.1, kappa=1.5, seed=7)
    assert np.allclose(result.mid, closes)


def test_max_inventory_is_respected():
    bars = _bars(_synthetic_closes(500, 1.0, seed=4))
    result = run_market_maker_on_bars(
        bars, gamma=1e-6, kappa=1.5, arrival_rate=0.3, max_inventory=3.0, seed=1,
    )
    assert np.max(np.abs(result.inventory)) <= 3.0


def test_higher_realized_volatility_widens_the_quoted_spread():
    # Same gamma/kappa/arrival_rate; only the input closes' volatility differs.
    calm = _bars(_synthetic_closes(300, sigma=0.2, seed=5))
    volatile = _bars(_synthetic_closes(300, sigma=2.0, seed=5))
    calm_result = run_market_maker_on_bars(calm, gamma=0.1, kappa=1.5, seed=7)
    volatile_result = run_market_maker_on_bars(volatile, gamma=0.1, kappa=1.5, seed=7)
    assert np.nanmean(volatile_result.ask - volatile_result.bid) > np.nanmean(calm_result.ask - calm_result.bid)


def test_result_is_compatible_with_markout_analysis():
    bars = _bars(_synthetic_closes(500, 1.0, seed=6))
    result = run_market_maker_on_bars(bars, gamma=0.1, kappa=1.5, arrival_rate=2.0, seed=7)
    assert result.fills  # sanity: this parameterization should produce fills
    markouts = compute_markouts(result, horizons=[1, 5])
    assert not markouts.empty
    assert np.allclose(
        markouts["total_markout_1"].to_numpy(),
        (markouts["spread_capture_1"] + markouts["adverse_selection_1"]).to_numpy(),
        equal_nan=True,
    )


def test_quote_band_reduces_requote_rate():
    bars = _bars(_synthetic_closes(2000, 1.0, seed=8))
    unbanded = run_market_maker_on_bars(bars, gamma=0.1, kappa=1.5, arrival_rate=2.0, seed=7)
    banded = run_market_maker_on_bars(bars, gamma=0.1, kappa=1.5, arrival_rate=2.0, quote_band=0.1, seed=7)
    assert summarize(banded)["bid_requote_rate"] < summarize(unbanded)["bid_requote_rate"]
