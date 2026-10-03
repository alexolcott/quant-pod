import numpy as np
import pytest

from quant_pod.marketmaking.order_flow import simulate_mid_price_path


def test_path_starts_at_mid0_and_has_right_length():
    path = simulate_mid_price_path(mid0=100.0, sigma=0.3, dt=1.0, n_steps=10, seed=1)
    assert path[0] == 100.0
    assert len(path) == 11


def test_path_is_deterministic_given_seed():
    a = simulate_mid_price_path(100.0, 0.3, 1.0, 500, seed=42)
    b = simulate_mid_price_path(100.0, 0.3, 1.0, 500, seed=42)
    assert np.array_equal(a, b)


def test_terminal_variance_matches_sigma_squared_times_horizon():
    sigma, dt, n_steps = 2.0, 1.0, 2000
    terminal = [simulate_mid_price_path(0.0, sigma, dt, n_steps, seed=s)[-1] for s in range(300)]
    sample_var = np.var(terminal)
    expected_var = sigma**2 * n_steps * dt
    assert sample_var == pytest.approx(expected_var, rel=0.25)
