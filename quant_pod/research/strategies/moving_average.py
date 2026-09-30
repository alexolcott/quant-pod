from __future__ import annotations

import pandas as pd

from quant_pod.research.strategy import Strategy


class MovingAverageCrossover(Strategy):
    """Long `position_size` shares when the fast MA is above the slow MA, flat otherwise."""

    strategy_id = "moving_average"

    def __init__(self, fast: int = 20, slow: int = 50, position_size: float = 100.0):
        if fast >= slow:
            raise ValueError("fast window must be smaller than slow window")
        self.fast = fast
        self.slow = slow
        self.position_size = position_size

    def generate_target_positions(self, bars: pd.DataFrame) -> pd.Series:
        fast_ma = bars["close"].rolling(self.fast).mean()
        slow_ma = bars["close"].rolling(self.slow).mean()
        long_signal = (fast_ma > slow_ma).fillna(False)
        return (long_signal * self.position_size).rename("target_position")
