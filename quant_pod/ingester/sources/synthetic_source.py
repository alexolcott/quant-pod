"""Synthetic GBM price generator.

Always works offline and deterministically (seeded by symbol), so the rest of
the pod can be developed/tested without depending on a flaky network source.
"""
from __future__ import annotations

import hashlib
from datetime import datetime

import numpy as np
import pandas as pd


class SyntheticSource:
    def __init__(self, annual_vol: float = 0.25, annual_drift: float = 0.08, start_price: float = 100.0):
        self.annual_vol = annual_vol
        self.annual_drift = annual_drift
        self.start_price = start_price

    def fetch(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        dates = pd.bdate_range(start=start, end=end, tz="UTC")
        n = len(dates)
        if n == 0:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        seed = int(hashlib.sha256(symbol.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)

        dt = 1 / 252
        drift = (self.annual_drift - 0.5 * self.annual_vol**2) * dt
        shocks = self.annual_vol * np.sqrt(dt) * rng.standard_normal(n)
        log_returns = drift + shocks
        close = self.start_price * np.exp(np.cumsum(log_returns))

        open_ = np.empty(n)
        open_[0] = self.start_price
        open_[1:] = close[:-1]

        intraday_range = np.abs(rng.standard_normal(n)) * self.annual_vol * np.sqrt(dt) * close
        high = np.maximum(open_, close) + intraday_range * 0.5
        low = np.minimum(open_, close) - intraday_range * 0.5
        volume = rng.integers(1_000_000, 10_000_000, size=n).astype(float)

        return pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=pd.DatetimeIndex(dates, name="timestamp"),
        )
