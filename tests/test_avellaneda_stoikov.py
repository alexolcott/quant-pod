import math

import numpy as np
import pytest

from quant_pod.marketmaking.avellaneda_stoikov import (
    AvellanedaStoikovQuoter,
    apply_tolerance_band,
    optimal_spread,
    reservation_price,
)


def test_reservation_price_equals_mid_when_flat():
    assert reservation_price(mid=100.0, inventory=0.0, gamma=0.1, sigma=0.3, time_remaining=0.5) == 100.0


def test_long_inventory_skews_reservation_price_down():
    r = reservation_price(mid=100.0, inventory=10.0, gamma=0.1, sigma=0.3, time_remaining=0.5)
    assert r < 100.0


def test_short_inventory_skews_reservation_price_up():
    r = reservation_price(mid=100.0, inventory=-10.0, gamma=0.1, sigma=0.3, time_remaining=0.5)
    assert r > 100.0


def test_spread_is_positive():
    assert optimal_spread(gamma=0.1, sigma=0.3, time_remaining=0.5, kappa=1.5) > 0


def test_spread_widens_as_time_remaining_grows():
    near_close = optimal_spread(gamma=0.1, sigma=0.3, time_remaining=0.01, kappa=1.5)
    far_from_close = optimal_spread(gamma=0.1, sigma=0.3, time_remaining=1.0, kappa=1.5)
    assert far_from_close > near_close


def test_spread_widens_with_higher_risk_aversion_when_inventory_risk_dominates():
    # optimal_spread is NOT globally monotonic in gamma -- the (2/gamma)*ln(1+gamma/kappa)
    # term shrinks as gamma grows even as the gamma*sigma^2*(T-t) term grows. Use a large
    # sigma/time_remaining so the inventory-risk term dominates and the net effect is a
    # widening spread, which is the regime this property actually holds in.
    low_gamma = optimal_spread(gamma=0.5, sigma=2.0, time_remaining=1.0, kappa=1.5)
    high_gamma = optimal_spread(gamma=2.0, sigma=2.0, time_remaining=1.0, kappa=1.5)
    assert high_gamma > low_gamma


def test_inventory_skew_magnitude_always_grows_with_risk_aversion():
    # Unlike the spread, the reservation-price skew for a fixed nonzero inventory
    # is linear in gamma, so this direction always holds.
    low = abs(reservation_price(mid=100.0, inventory=10.0, gamma=0.05, sigma=0.3, time_remaining=0.5) - 100.0)
    high = abs(reservation_price(mid=100.0, inventory=10.0, gamma=0.5, sigma=0.3, time_remaining=0.5) - 100.0)
    assert high > low


def test_risk_neutral_limit_matches_closed_form():
    # As gamma -> 0, spread -> 2/kappa and the inventory skew term vanishes.
    spread = optimal_spread(gamma=1e-6, sigma=0.3, time_remaining=0.5, kappa=1.5)
    assert math.isclose(spread, 2.0 / 1.5, rel_tol=1e-3)
    assert optimal_spread(gamma=0.0, sigma=0.3, time_remaining=0.5, kappa=1.5) == pytest.approx(2.0 / 1.5)


def test_quoter_centers_on_reservation_price():
    quoter = AvellanedaStoikovQuoter(gamma=0.1, kappa=1.5, sigma=0.3)
    bid, ask = quoter.quote(mid=100.0, inventory=0.0, time_remaining=0.5)
    assert bid < 100.0 < ask
    assert ask - bid == pytest.approx(optimal_spread(0.1, 0.3, 0.5, 1.5))


def test_tolerance_band_adopts_immediately_on_first_call():
    bid, ask = apply_tolerance_band(99.0, 101.0, posted_bid=None, posted_ask=None, band=0.5)
    assert (bid, ask) == (99.0, 101.0)


def test_tolerance_band_holds_a_small_drift():
    bid, ask = apply_tolerance_band(99.1, 101.1, posted_bid=99.0, posted_ask=101.0, band=0.5)
    assert (bid, ask) == (99.0, 101.0)  # drift of 0.1 is within the 0.5 band -- stays put


def test_tolerance_band_moves_on_a_large_drift():
    bid, ask = apply_tolerance_band(99.8, 101.8, posted_bid=99.0, posted_ask=101.0, band=0.5)
    assert (bid, ask) == (99.8, 101.8)  # drift of 0.8 exceeds the 0.5 band -- repost


def test_tolerance_band_moves_exactly_at_the_boundary_is_still_held():
    # Strictly greater-than: a drift exactly equal to the band does NOT trigger a requote.
    bid, _ = apply_tolerance_band(99.5, 101.0, posted_bid=99.0, posted_ask=101.0, band=0.5)
    assert bid == 99.0


def test_tolerance_band_adopts_a_suppressed_side_immediately():
    # theoretical NaN (e.g. pulled by a max_inventory limit) -- never sticky.
    bid, ask = apply_tolerance_band(np.nan, 101.1, posted_bid=99.0, posted_ask=101.0, band=0.5)
    assert np.isnan(bid)
    assert ask == 101.0  # the untouched side is unaffected


def test_tolerance_band_recovers_from_a_previously_suppressed_side_immediately():
    # posted NaN (was suppressed last tick) with a fresh finite theoretical value --
    # adopted immediately regardless of band, not compared against NaN.
    bid, _ = apply_tolerance_band(99.0, 101.0, posted_bid=np.nan, posted_ask=101.0, band=0.5)
    assert bid == 99.0
