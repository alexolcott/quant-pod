import numpy as np
import pandas as pd
import pytest

from quant_pod.research.backtester import run_backtest
from quant_pod.research.strategies.moving_average import MovingAverageCrossover
from quant_pod.research.strategy import Strategy


def make_trending_bars(n: int = 120) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-01", periods=n, tz="UTC")
    close = 100 + np.arange(n) * 0.5  # steadily rising
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1_000_000.0},
        index=pd.DatetimeIndex(dates, name="timestamp"),
    )


class AlwaysLongStrategy(Strategy):
    strategy_id = "always_long"

    def generate_target_positions(self, bars: pd.DataFrame) -> pd.Series:
        return pd.Series(100.0, index=bars.index)


class AlwaysFlatStrategy(Strategy):
    strategy_id = "always_flat"

    def generate_target_positions(self, bars: pd.DataFrame) -> pd.Series:
        return pd.Series(0.0, index=bars.index)


def test_always_flat_strategy_has_no_trades_and_constant_equity():
    bars = make_trending_bars()
    result = run_backtest(AlwaysFlatStrategy(), bars)
    assert len(result.trades) == 0
    assert result.equity_curve.nunique() == 1


def test_always_long_strategy_profits_in_uptrend():
    bars = make_trending_bars()
    result = run_backtest(AlwaysLongStrategy(), bars, initial_cash=100_000)
    assert result.equity_curve.iloc[-1] > result.equity_curve.iloc[0]
    assert len(result.trades) == 1  # one entry, never exits


def test_backtest_has_no_lookahead_bias():
    """A strategy targeting position based on bar N's close must not be filled
    at bar N's own open -- it should trade at N+1's open."""
    bars = make_trending_bars(10)

    class JumpAtBar5(Strategy):
        strategy_id = "jump"

        def generate_target_positions(self, bars: pd.DataFrame) -> pd.Series:
            targets = pd.Series(0.0, index=bars.index)
            targets.iloc[5:] = 100.0
            return targets

    result = run_backtest(JumpAtBar5(), bars)
    trade_ts = result.trades.iloc[0]["timestamp"]
    assert trade_ts == bars.index[6]  # signal known after bar 5 closes -> trades at bar 6's open


def test_moving_average_crossover_requires_fast_less_than_slow():
    with pytest.raises(ValueError):
        MovingAverageCrossover(fast=50, slow=20)
