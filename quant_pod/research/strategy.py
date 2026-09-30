"""Strategy base class.

The same subclass instance is used by both the vectorized `backtester` (for
fast research iteration) and the live `trader.engine` (for the real event
loop), so a strategy's logic only needs to be written once.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Subclasses implement `generate_target_positions`, mapping a history of
    bars to a target position (in shares) for each timestamp.
    """

    strategy_id: str = "unnamed"

    @abstractmethod
    def generate_target_positions(self, bars: pd.DataFrame) -> pd.Series:
        """`bars` is indexed by timestamp with columns open/high/low/close/volume,
        covering all history available up to "now". Returns a Series of target
        share positions aligned to `bars.index` (vectorized/backtest use), or
        callers may call `.iloc[-1]` for the latest single target (live use).
        """
        raise NotImplementedError

    def target_position_for(self, bars_so_far: pd.DataFrame) -> float:
        """Convenience for the live trader: target position given history up to now."""
        targets = self.generate_target_positions(bars_so_far)
        if targets.empty:
            return 0.0
        return float(targets.iloc[-1])
