"""Performance metrics shared by backtest reports and live trader reports."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def cagr(equity_curve: pd.Series) -> float:
    if len(equity_curve) < 2:
        return 0.0
    n_years = len(equity_curve) / TRADING_DAYS_PER_YEAR
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    if total_return <= 0 or n_years <= 0:
        return -1.0
    return total_return ** (1 / n_years) - 1


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    excess = returns - risk_free / TRADING_DAYS_PER_YEAR
    std = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    excess = returns - risk_free / TRADING_DAYS_PER_YEAR
    downside = excess[excess < 0]
    downside_std = downside.std()
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return float(drawdown.min())


def calmar_ratio(equity_curve: pd.Series) -> float:
    mdd = abs(max_drawdown(equity_curve))
    if mdd == 0:
        return 0.0
    return cagr(equity_curve) / mdd


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical Value-at-Risk at `confidence`, as a positive loss fraction."""
    if returns.empty:
        return 0.0
    return float(-np.quantile(returns.dropna(), 1 - confidence))


def turnover(trades: pd.DataFrame, avg_equity: float) -> float:
    if trades.empty or avg_equity == 0:
        return 0.0
    gross_traded = (trades["quantity"].abs() * trades["price"]).sum()
    return float(gross_traded / avg_equity)


def summarize(equity_curve: pd.Series, returns: pd.Series, trades: pd.DataFrame) -> dict:
    return {
        "total_return": float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) if len(equity_curve) else 0.0,
        "cagr": cagr(equity_curve),
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "max_drawdown": max_drawdown(equity_curve),
        "calmar": calmar_ratio(equity_curve),
        "var_95": historical_var(returns, 0.95),
        "turnover": turnover(trades, equity_curve.mean() if len(equity_curve) else 0.0),
        "num_trades": len(trades),
    }
