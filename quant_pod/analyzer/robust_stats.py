"""Backtest-validation statistics that go beyond a single reported Sharpe ratio:

- Probabilistic / Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2012/2014):
  how likely is the observed Sharpe to be genuine skill rather than noise, and
  how much of it survives once you account for how many strategy variants/
  symbols you tried before picking this one?
- Block-bootstrap confidence intervals on the Sharpe ratio, for returns that
  aren't i.i.d. (autocorrelated P&L, as almost all strategy returns are).
- Purged & embargoed walk-forward splits, so cross-validation doesn't leak
  information across the train/test boundary.

All Sharpe-ratio math here operates in **per-period** units (mean/std of the
raw returns series, not annualized) because the statistical tests are only
valid in the same units as the sample size `T` they're parameterized by;
annualization is applied only when a function returns a value meant for
human-facing reporting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329


def per_period_sharpe(returns: pd.Series, risk_free: float = 0.0) -> float:
    excess = returns - risk_free
    std = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std)


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """P(true Sharpe > benchmark_sharpe | observed_sharpe, n_obs samples of the
    given skew/kurtosis). `kurtosis` is the raw (Pearson) kurtosis -- 3.0 for a
    normal distribution, not excess kurtosis.
    """
    if n_obs < 2:
        return 0.5
    denom = 1 - skew * observed_sharpe + ((kurtosis - 1) / 4) * observed_sharpe**2
    denom = max(denom, 1e-12)
    z = (observed_sharpe - benchmark_sharpe) * np.sqrt(n_obs - 1) / np.sqrt(denom)
    return float(norm.cdf(z))


def expected_max_sharpe_under_null(n_trials: int, sharpe_std: float) -> float:
    """E[max of `n_trials` independent Sharpe-ratio estimates, each ~ N(0,
    sharpe_std^2)] -- the benchmark a selected Sharpe ratio must clear to not
    be explainable by multiple-testing luck alone.

    With a single trial there's nothing to deflate against, so this returns 0.
    """
    if n_trials <= 1 or sharpe_std <= 0:
        return 0.0
    z1 = norm.ppf(1 - 1 / n_trials)
    z2 = norm.ppf(1 - 1 / (n_trials * np.e))
    return float(sharpe_std * ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2))


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int,
    trial_sharpe_std: float | None = None,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> dict:
    """The Deflated Sharpe Ratio: PSR evaluated against the expected best
    Sharpe ratio you'd see from `n_trials` independent no-skill variants, i.e.
    how much of the observed Sharpe survives after accounting for how many
    strategies/parameter sets/symbols were tried before this one was picked.

    `trial_sharpe_std` is the cross-sectional standard deviation of the
    trials' Sharpe ratios -- pass the actual empirical std across your N
    trials when you have it (most rigorous); omitted, it falls back to the
    per-period Sharpe estimator's standard error under the null of zero skill,
    1/sqrt(n_obs - 1).
    """
    n_obs = len(returns)
    sharpe_hat = per_period_sharpe(returns, risk_free)
    skew = float(returns.skew()) if n_obs > 2 else 0.0
    kurtosis = float(returns.kurt()) + 3.0 if n_obs > 3 else 3.0  # pandas .kurt() is excess kurtosis

    if trial_sharpe_std is None:
        trial_sharpe_std = 1.0 / np.sqrt(n_obs - 1) if n_obs > 1 else 0.0
    benchmark_sharpe = expected_max_sharpe_under_null(n_trials, trial_sharpe_std)

    dsr = probabilistic_sharpe_ratio(sharpe_hat, benchmark_sharpe, n_obs, skew, kurtosis)
    return {
        "deflated_sharpe_ratio": dsr,
        "observed_sharpe_annualized": sharpe_hat * np.sqrt(periods_per_year),
        "benchmark_sharpe_annualized": benchmark_sharpe * np.sqrt(periods_per_year),
        "n_trials": n_trials,
        "n_obs": n_obs,
        "skew": skew,
        "kurtosis": kurtosis,
    }


def block_bootstrap_sharpe_ci(
    returns: pd.Series,
    n_bootstrap: int = 1000,
    block_size: int = 20,
    confidence: float = 0.95,
    periods_per_year: int = 252,
    seed: int | None = None,
) -> dict:
    """Confidence interval on the annualized Sharpe ratio via the (circular)
    moving-block bootstrap -- resamples contiguous blocks rather than
    individual observations, so it doesn't assume returns are i.i.d.
    """
    values = returns.to_numpy()
    n = len(values)
    if n == 0:
        raise ValueError("returns is empty")
    block_size = max(min(block_size, n), 1)

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    boot_sharpes = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        starts = rng.integers(0, n, size=n_blocks)  # circular: wraps past the end
        sample = np.concatenate([np.take(values, np.arange(s, s + block_size), mode="wrap") for s in starts])[:n]
        mean, std = sample.mean(), sample.std(ddof=1) if n > 1 else 0.0
        boot_sharpes[i] = 0.0 if std == 0 else (mean / std) * np.sqrt(periods_per_year)

    lower_q, upper_q = (1 - confidence) / 2, 1 - (1 - confidence) / 2
    lo, hi = np.quantile(boot_sharpes, [lower_q, upper_q])
    return {
        "point_estimate": per_period_sharpe(returns) * np.sqrt(periods_per_year),
        "lower": float(lo),
        "upper": float(hi),
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
        "block_size": block_size,
    }


def purged_walk_forward_splits(n_samples: int, n_splits: int, embargo: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window walk-forward splits: fold k trains on everything
    before the test block, with the `embargo` samples immediately preceding
    the test block purged from training -- guards against leakage when a
    training-set label/feature window straddles the train/test boundary
    (Lopez de Prado, *Advances in Financial Machine Learning*, ch. 7).

    Returns `n_splits` (train_idx, test_idx) pairs; test blocks are
    contiguous and non-overlapping, covering the final `n_splits / (n_splits
    + 1)` fraction of the data (the first fold always needs an initial
    training block, so it can't start at sample 0).
    """
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    fold_size = n_samples // (n_splits + 1)
    if fold_size < 1:
        raise ValueError(f"not enough samples ({n_samples}) for {n_splits} fold(s)")

    splits = []
    for k in range(1, n_splits + 1):
        test_start = k * fold_size
        test_end = n_samples if k == n_splits else (k + 1) * fold_size
        train_end = max(test_start - embargo, 0)
        splits.append((np.arange(0, train_end), np.arange(test_start, test_end)))
    return splits
