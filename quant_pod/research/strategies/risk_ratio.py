"""Composite risk-ratio strategy, in the spirit of the standalone Risk Metric
project: combine rolling Sharpe, historical VaR, and max drawdown into one
score, and buy when the score signals the stock is oversold relative to its
own risk profile.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant_pod.research.strategy import Strategy


class RiskRatioStrategy(Strategy):
    strategy_id = "risk_ratio"

    def __init__(
        self,
        window: int = 60,
        var_confidence: float = 0.95,
        buy_threshold: float = -1.0,
        position_size: float = 100.0,
    ):
        self.window = window
        self.var_confidence = var_confidence
        self.buy_threshold = buy_threshold
        self.position_size = position_size

    def _risk_ratio(self, returns: pd.Series, close: pd.Series) -> pd.Series:
        w = self.window
        rolling_mean = returns.rolling(w).mean()
        rolling_std = returns.rolling(w).std()
        sharpe = (rolling_mean / rolling_std) * np.sqrt(252)

        var = returns.rolling(w).quantile(1 - self.var_confidence)

        rolling_max = close.rolling(w).max()
        drawdown = (close - rolling_max) / rolling_max

        # Normalize each component to a z-score over the window so they're
        # comparable before combining; higher combined score = healthier risk profile.
        def zscore(s: pd.Series) -> pd.Series:
            return (s - s.rolling(w).mean()) / s.rolling(w).std()

        return zscore(sharpe) + zscore(var) + zscore(drawdown)

    def generate_target_positions(self, bars: pd.DataFrame) -> pd.Series:
        returns = bars["close"].pct_change()
        ratio = self._risk_ratio(returns, bars["close"])
        buy_signal = (ratio < self.buy_threshold).fillna(False)
        return (buy_signal * self.position_size).rename("target_position")
