import numpy as np
import pandas as pd
import pytest

from quant_pod.analyzer import metrics as m


def test_max_drawdown_simple():
    equity = pd.Series([100, 120, 90, 110])
    assert m.max_drawdown(equity) == pytest.approx(90 / 120 - 1)


def test_sharpe_ratio_zero_when_no_variance():
    returns = pd.Series([0.0] * 10)
    assert m.sharpe_ratio(returns) == 0.0


def test_sharpe_ratio_positive_for_consistently_positive_returns():
    returns = pd.Series([0.01] * 50)
    # constant positive returns have zero std -> sharpe defined as 0 by our guard
    assert m.sharpe_ratio(returns) == 0.0


def test_sharpe_ratio_sign_matches_return_direction():
    rng = np.random.default_rng(0)
    good_returns = pd.Series(0.001 + 0.01 * rng.standard_normal(500))
    bad_returns = pd.Series(-0.001 + 0.01 * rng.standard_normal(500))
    assert m.sharpe_ratio(good_returns) > m.sharpe_ratio(bad_returns)


def test_cagr_matches_known_growth():
    equity = pd.Series([100.0] + [100.0] * 251 + [200.0])  # doubles over ~1 year (252 trading days)
    assert m.cagr(equity) > 0.9  # should be close to 100% annualized growth


def test_historical_var_is_positive_loss_fraction():
    returns = pd.Series(np.linspace(-0.05, 0.05, 100))
    var = m.historical_var(returns, confidence=0.95)
    assert var > 0


def test_turnover_zero_with_no_trades():
    trades = pd.DataFrame(columns=["quantity", "price"])
    assert m.turnover(trades, avg_equity=100_000) == 0.0
