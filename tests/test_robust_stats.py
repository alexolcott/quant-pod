import numpy as np
import pandas as pd
import pytest

from quant_pod.analyzer.robust_stats import (
    block_bootstrap_sharpe_ci,
    deflated_sharpe_ratio,
    expected_max_sharpe_under_null,
    probabilistic_sharpe_ratio,
    purged_walk_forward_splits,
)


def _synthetic_returns(n: int, drift: float, vol: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(drift, vol, size=n))


# --- probabilistic_sharpe_ratio ------------------------------------------------

def test_psr_is_half_when_observed_equals_benchmark():
    assert probabilistic_sharpe_ratio(0.05, 0.05, n_obs=500) == pytest.approx(0.5)


def test_psr_increases_with_observed_sharpe():
    low = probabilistic_sharpe_ratio(0.0, 0.0, n_obs=500)
    high = probabilistic_sharpe_ratio(0.1, 0.0, n_obs=500)
    assert high > low


def test_psr_increases_with_more_observations_for_the_same_edge():
    few_obs = probabilistic_sharpe_ratio(0.05, 0.0, n_obs=30)
    many_obs = probabilistic_sharpe_ratio(0.05, 0.0, n_obs=3000)
    assert many_obs > few_obs  # more data makes the same observed edge more convincing


# --- expected_max_sharpe_under_null --------------------------------------------

def test_single_trial_has_no_benchmark():
    assert expected_max_sharpe_under_null(1, sharpe_std=0.1) == 0.0


def test_benchmark_grows_with_more_trials():
    few = expected_max_sharpe_under_null(2, sharpe_std=0.1)
    many = expected_max_sharpe_under_null(1000, sharpe_std=0.1)
    assert 0.0 < few < many


# --- deflated_sharpe_ratio ------------------------------------------------------

def test_dsr_falls_as_trial_count_rises_for_fixed_returns():
    returns = _synthetic_returns(1000, drift=0.0006, vol=0.01, seed=1)
    single_trial = deflated_sharpe_ratio(returns, n_trials=1)
    many_trials = deflated_sharpe_ratio(returns, n_trials=1000)
    assert single_trial["deflated_sharpe_ratio"] > many_trials["deflated_sharpe_ratio"]


def test_dsr_is_bounded_in_unit_interval():
    returns = _synthetic_returns(500, drift=0.0003, vol=0.02, seed=2)
    result = deflated_sharpe_ratio(returns, n_trials=50)
    assert 0.0 <= result["deflated_sharpe_ratio"] <= 1.0


# --- block_bootstrap_sharpe_ci ---------------------------------------------------

def test_bootstrap_ci_is_deterministic_given_seed():
    returns = _synthetic_returns(300, drift=0.0005, vol=0.01, seed=3)
    a = block_bootstrap_sharpe_ci(returns, n_bootstrap=200, block_size=10, seed=99)
    b = block_bootstrap_sharpe_ci(returns, n_bootstrap=200, block_size=10, seed=99)
    assert a == b


def test_bootstrap_ci_lower_bound_below_upper_bound():
    returns = _synthetic_returns(300, drift=0.0005, vol=0.01, seed=3)
    ci = block_bootstrap_sharpe_ci(returns, n_bootstrap=200, block_size=10, seed=1)
    assert ci["lower"] < ci["upper"]


def test_bootstrap_ci_narrows_with_more_data():
    short = _synthetic_returns(100, drift=0.0005, vol=0.01, seed=4)
    long = _synthetic_returns(4000, drift=0.0005, vol=0.01, seed=4)
    ci_short = block_bootstrap_sharpe_ci(short, n_bootstrap=500, block_size=10, seed=1)
    ci_long = block_bootstrap_sharpe_ci(long, n_bootstrap=500, block_size=10, seed=1)
    assert (ci_long["upper"] - ci_long["lower"]) < (ci_short["upper"] - ci_short["lower"])


# --- purged_walk_forward_splits ---------------------------------------------------

def test_splits_respect_the_embargo_gap():
    splits = purged_walk_forward_splits(n_samples=1000, n_splits=4, embargo=10)
    for train_idx, test_idx in splits:
        if len(train_idx) == 0:
            continue
        assert test_idx[0] - train_idx[-1] > 10


def test_test_blocks_are_contiguous_and_non_overlapping():
    splits = purged_walk_forward_splits(n_samples=1000, n_splits=4, embargo=0)
    all_test_idx = np.concatenate([test_idx for _, test_idx in splits])
    assert len(all_test_idx) == len(set(all_test_idx.tolist()))  # no overlap
    assert np.array_equal(all_test_idx, np.sort(all_test_idx))  # strictly increasing, i.e. in fold order


def test_too_many_splits_for_the_sample_size_raises():
    with pytest.raises(ValueError):
        purged_walk_forward_splits(n_samples=3, n_splits=10, embargo=0)
