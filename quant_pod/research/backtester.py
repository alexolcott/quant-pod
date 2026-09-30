"""Vectorized backtest engine for fast strategy iteration.

Takes a Strategy + a bar history and produces a trade log and equity curve by
turning target positions into next-bar fills (no lookahead: today's signal
trades at tomorrow's open) with a simple flat commission + slippage model.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_pod.research.strategy import Strategy


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    returns: pd.Series


def run_backtest(
    strategy: Strategy,
    bars: pd.DataFrame,
    initial_cash: float = 100_000.0,
    commission_per_trade: float = 1.0,
    slippage_bps: float = 1.0,
) -> BacktestResult:
    targets = strategy.generate_target_positions(bars).reindex(bars.index).fillna(0.0)
    # Trade at the next bar's open to avoid lookahead bias.
    positions = targets.shift(1).fillna(0.0)

    trade_qty = positions.diff().fillna(positions.iloc[0] if len(positions) else 0.0)
    fill_price = bars["open"] * (1 + (slippage_bps / 10_000) * trade_qty.apply(lambda q: 1 if q > 0 else (-1 if q < 0 else 0)))

    trade_mask = trade_qty != 0
    trades = pd.DataFrame(
        {
            "timestamp": bars.index[trade_mask],
            "quantity": trade_qty[trade_mask].values,
            "price": fill_price[trade_mask].values,
        }
    ).reset_index(drop=True)
    trades["commission"] = commission_per_trade

    cash = initial_cash - (trade_qty * fill_price).cumsum() - trade_mask.cumsum() * commission_per_trade
    holdings_value = positions * bars["close"]
    equity_curve = (cash + holdings_value).rename("equity")

    returns = equity_curve.pct_change().fillna(0.0).rename("returns")

    return BacktestResult(equity_curve=equity_curve, positions=positions, trades=trades, returns=returns)
