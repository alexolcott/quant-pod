import math

import pytest

from quant_pod.marketmaking.avellaneda_stoikov import (
    AvellanedaStoikovQuoter,
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
