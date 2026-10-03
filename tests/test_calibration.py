import numpy as np
import pandas as pd
import pytest

from quant_pod.marketmaking.avellaneda_stoikov import optimal_spread
from quant_pod.marketmaking.calibration import (
    arrival_rate_from_volume,
    calibrate_kappa_for_target_spread,
    garman_klass_sigma,
)


def _flat_bars(n: int, price: float, volume: float = 1_000_000.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": price, "high": price, "low": price, "close": price, "volume": volume}, index=idx,
    )


def _noisy_bars(n: int, price0: float, daily_pct_vol: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    closes = price0 * np.cumprod(1 + rng.normal(0, daily_pct_vol, size=n))
    opens = np.concatenate([[price0], closes[:-1]])
    highs = np.maximum(opens, closes) * (1 + rng.uniform(0, daily_pct_vol, size=n))
    lows = np.minimum(opens, closes) * (1 - rng.uniform(0, daily_pct_vol, size=n))
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": 1_000_000.0}, index=idx,
    )


# --- garman_klass_sigma ---------------------------------------------------

def test_zero_movement_bars_give_zero_sigma():
    bars = _flat_bars(30, price=100.0)
    sigma = garman_klass_sigma(bars, window=10)
    assert (sigma.dropna() == 0.0).all()


def test_sigma_is_never_negative():
    bars = _noisy_bars(100, price0=50.0, daily_pct_vol=0.03, seed=1)
    sigma = garman_klass_sigma(bars, window=10)
    assert (sigma.dropna() >= 0.0).all()


def test_sigma_scales_with_price_level():
    # Same percentage moves at 10x the price level should give ~10x the dollar sigma --
    # this is the exact bug the naive closes.diff().std() estimator had (see module docstring).
    bars_low = _noisy_bars(200, price0=50.0, daily_pct_vol=0.02, seed=7)
    bars_high = bars_low.copy()
    bars_high[["open", "high", "low", "close"]] *= 10

    sigma_low = garman_klass_sigma(bars_low, window=20).dropna().mean()
    sigma_high = garman_klass_sigma(bars_high, window=20).dropna().mean()
    assert sigma_high == pytest.approx(sigma_low * 10, rel=1e-6)


def test_sigma_has_nan_before_the_window_fills():
    bars = _noisy_bars(50, price0=100.0, daily_pct_vol=0.02, seed=2)
    sigma = garman_klass_sigma(bars, window=20)
    assert sigma.iloc[:19].isna().all()
    assert sigma.iloc[19:].notna().all()


# --- arrival_rate_from_volume ---------------------------------------------

def test_higher_volume_gives_higher_arrival_rate():
    low_vol = _flat_bars(30, price=100.0, volume=1_000_000)
    high_vol = _flat_bars(30, price=100.0, volume=10_000_000)
    a_low = arrival_rate_from_volume(low_vol, fill_size=100.0, window=10).dropna().iloc[-1]
    a_high = arrival_rate_from_volume(high_vol, fill_size=100.0, window=10).dropna().iloc[-1]
    assert a_high > a_low


def test_arrival_rate_inversely_proportional_to_fill_size():
    bars = _flat_bars(30, price=100.0, volume=1_000_000)
    a_small = arrival_rate_from_volume(bars, fill_size=10.0, window=10).dropna().iloc[-1]
    a_large = arrival_rate_from_volume(bars, fill_size=100.0, window=10).dropna().iloc[-1]
    assert a_small == pytest.approx(a_large * 10, rel=1e-9)


def test_arrival_rate_scales_with_participation_rate():
    bars = _flat_bars(30, price=100.0, volume=1_000_000)
    a_low = arrival_rate_from_volume(bars, fill_size=100.0, window=10, participation_rate=0.001).dropna().iloc[-1]
    a_high = arrival_rate_from_volume(bars, fill_size=100.0, window=10, participation_rate=0.01).dropna().iloc[-1]
    assert a_high == pytest.approx(a_low * 10, rel=1e-9)


# --- calibrate_kappa_for_target_spread ------------------------------------

def test_calibrated_kappa_reproduces_the_target_spread():
    gamma, sigma, time_horizon = 0.1, 1.5, 1.0
    target = 0.5
    kappa = calibrate_kappa_for_target_spread(target, gamma, sigma, time_horizon)
    assert optimal_spread(gamma, sigma, time_horizon, kappa) == pytest.approx(target, rel=1e-9)


def test_calibrated_kappa_reproduces_a_different_target_spread():
    gamma, sigma, time_horizon = 0.05, 2.0, 2.0
    target = 3.0
    kappa = calibrate_kappa_for_target_spread(target, gamma, sigma, time_horizon)
    assert optimal_spread(gamma, sigma, time_horizon, kappa) == pytest.approx(target, rel=1e-9)


def test_wider_target_spread_gives_smaller_kappa():
    gamma, sigma, time_horizon = 0.1, 1.0, 1.0
    tight = calibrate_kappa_for_target_spread(0.3, gamma, sigma, time_horizon)
    wide = calibrate_kappa_for_target_spread(1.0, gamma, sigma, time_horizon)
    assert wide < tight


def test_target_spread_below_inventory_risk_floor_raises():
    gamma, sigma, time_horizon = 0.5, 2.0, 1.0  # inventory-risk floor = 0.5*4*1 = 2.0
    with pytest.raises(ValueError):
        calibrate_kappa_for_target_spread(1.0, gamma, sigma, time_horizon)
