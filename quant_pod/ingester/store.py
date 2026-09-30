"""Local Parquet-backed store for ingested bars, partitioned by symbol."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from quant_pod.common.config import DATA_DIR


def _path_for(symbol: str) -> Path:
    return DATA_DIR / f"{symbol.upper()}.parquet"


def write_bars(symbol: str, df: pd.DataFrame) -> int:
    """Merge `df` (indexed by UTC timestamp, columns open/high/low/close/volume)
    into the existing store for `symbol`, deduping on timestamp. Returns row count.
    """
    path = _path_for(symbol)
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, df])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = df.sort_index()

    combined.to_parquet(path)
    return len(combined)


def read_bars(symbol: str, start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
    path = _path_for(symbol)
    if not path.exists():
        raise FileNotFoundError(
            f"No stored data for {symbol!r}. Run: quant-pod ingest fetch {symbol} --start ... --end ..."
        )
    df = pd.read_parquet(path)
    if start is not None:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end, tz="UTC")]
    return df


def has_symbol(symbol: str) -> bool:
    return _path_for(symbol).exists()
