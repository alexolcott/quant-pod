"""Free historical OHLCV data via yfinance. No API key required."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import yfinance as yf


class YFinanceSource:
    def fetch(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.rename(
            columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
        )[["open", "high", "low", "close", "volume"]]
        df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
        df.index.name = "timestamp"
        return df
