"""Calibrating the Avellaneda-Stoikov parameters from real data, instead of
the illustrative defaults used elsewhere in this package.

Be upfront about the real limit here: a textbook calibration of `kappa`
(fitting the fill-intensity-vs-distance curve lambda(delta) = A*exp(-kappa*delta)
to observed data) needs quote-level or trade-level data -- who got hit at what
distance from the touch, how often. This repo only has daily OHLCV bars. So:

- `sigma` -- genuinely, properly calibrated here. Garman-Klass is a real,
  textbook-standard volatility estimator that's strictly more statistically
  efficient than a naive close-to-close diff for the same window, and it's
  computed in log-return space and converted to dollar units at the current
  price level -- which also fixes a real bug found earlier in this project:
  `historical.py`'s original `closes.diff().std()` estimator used raw dollar
  diffs, so its output silently drifted with the stock's price level instead
  of tracking actual volatility (see docs/MARKET_MAKING.md's AAPL
  calm-vs-volatile comparison).
- `arrival_rate` (A) -- calibrated as a genuine proxy: real traded volume,
  scaled by an assumed participation rate (what fraction of total volume a
  market maker this size would realistically capture) and `fill_size`. This
  ties A to a real, observable number instead of an arbitrary illustrative
  constant, but the participation rate itself is a judgment call, not fit to
  anything -- it's clearly a parameter, not a result.
- `kappa` -- NOT independently calibrated here, because the data to do that
  honestly doesn't exist in this repo. Instead, `calibrate_kappa_for_target_spread`
  inverts the optimal-spread formula to find the kappa that makes the model's
  own quoted spread match a target (e.g. a typical real NBBO spread for a
  large-cap name). This is a real technique used when you don't have a fill
  curve but do have a sense of where the market actually quotes -- but it's
  explicitly a proxy for calibration, not a measurement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def garman_klass_sigma(bars: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling dollar-volatility estimate using the Garman-Klass (1980)
    OHLC estimator, which uses the full high/low/open/close range each bar
    instead of just the close-to-close move -- lower variance than a
    close-to-close estimator for the same window, a standard result in the
    volatility-estimation literature. Computed in log-return space, then
    converted to dollar units by scaling by that bar's close -- so the
    output tracks actual volatility as the stock's price level changes,
    unlike a raw price-diff std.

    Known limitation: Garman-Klass assumes continuous trading within the
    bar and no overnight jump -- real daily bars have both (the close-to-open
    gap is ignored entirely). It will understate true volatility on names
    with large overnight moves (earnings, gaps).
    """
    log_hl = np.log(bars["high"] / bars["low"])
    log_co = np.log(bars["close"] / bars["open"])
    per_bar_variance = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    rolling_variance = per_bar_variance.rolling(window).mean()
    return (np.sqrt(rolling_variance.clip(lower=0)) * bars["close"]).rename("sigma")


def arrival_rate_from_volume(
    bars: pd.DataFrame,
    fill_size: float,
    window: int = 20,
    participation_rate: float = 0.005,
) -> pd.Series:
    """Rolling estimate of `arrival_rate` (A) from real traded volume: average
    daily volume over `window` bars, times the assumed `participation_rate`
    (the fraction of all volume a market maker this size would realistically
    capture -- 0.5% by default, a judgment call, not a fit), divided by
    `fill_size` to convert shares/bar into "arrivals of size `fill_size`"/bar.

    This is a genuine tie to real trading activity (a thinly-traded name gets
    a much lower A than AAPL), but `participation_rate` is an assumption you
    should adjust for the actual competitive landscape, not a calibrated
    constant.

    Important: this is calibrated for the simulated day's worth of real
    volume -- feeding it into `historical.run_market_maker_on_bars` or the
    live engine (both use dt=1 per bar) means `arrival_rate * dt` is the same
    huge number, which saturates `1 - exp(-A*dt)` to ~1.0 for any liquid
    name: a certainty of at least one fill per side per day, not a
    probability. That's not wrong, exactly -- a touch-level AAPL quote really
    would get hit within a day -- but it exposes that "one fill-or-no-fill
    roll per side per bar" is too coarse a resolution to represent real
    large-cap liquidity at all; this value is architecturally suited to a
    small `dt` (the synthetic simulator's per-second ticks), not a daily bar.
    """
    avg_volume = bars["volume"].rolling(window).mean()
    return (avg_volume * participation_rate / fill_size).rename("arrival_rate")


def calibrate_kappa_for_target_spread(
    target_spread: float,
    gamma: float,
    sigma: float,
    time_horizon: float,
) -> float:
    """Inverts `avellaneda_stoikov.optimal_spread` to find the kappa that
    makes the model quote `target_spread` (in the same price units as
    `sigma`) at zero inventory. A substitute for calibrating kappa from a
    real fill-intensity curve when you don't have the data for that but do
    have a sense of where the market actually quotes (e.g. a typical NBBO
    spread for this name's liquidity tier).

    Raises ValueError if `target_spread` is tighter than the inventory-risk
    floor `gamma * sigma^2 * time_horizon` alone -- no kappa can quote that
    tight, since the order-arrival term only ever adds width.
    """
    if gamma <= 0:
        raise ValueError("gamma must be positive to calibrate kappa")
    inventory_risk_floor = gamma * sigma**2 * time_horizon
    if target_spread <= inventory_risk_floor:
        raise ValueError(
            f"target_spread ({target_spread:.4f}) must exceed the inventory-risk "
            f"floor gamma*sigma^2*time_horizon ({inventory_risk_floor:.4f}) -- "
            "no kappa can quote tighter than that term alone."
        )
    arrival_term = (target_spread - inventory_risk_floor) * gamma / 2
    return gamma / (np.expm1(arrival_term))
