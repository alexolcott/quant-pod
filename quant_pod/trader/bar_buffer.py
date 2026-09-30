"""Amortized-O(1)-append buffer for streaming bars into a DataFrame.

The naive way to do this -- append each Bar to a `list[dict]` and call
`pd.DataFrame(bar_rows)` every tick -- rebuilds the DataFrame from scratch on
every call: pandas has to walk the list of dicts, infer a dtype per column,
and allocate fresh arrays, and it pays that cost again on every single tick
even though almost all of the data hasn't changed since the last one.

This keeps pre-allocated, fixed-dtype numpy arrays (doubling capacity when
full, like a Python list does internally) so `append` is O(1) amortized, and
`as_dataframe()` wraps zero-copy views of the filled portion -- no dtype
inference, no per-row Python-level work.

Timestamps are stored tz-naive (UTC is assumed, matching the ingester's
store and replay feed) -- fine here since neither example strategy's
`.rolling(...)` call is time-based, only index-position-based.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant_pod.common.events import Bar

_FIELDS = ("open", "high", "low", "close", "volume")


class BarBuffer:
    def __init__(self, initial_capacity: int = 1024):
        self._capacity = max(initial_capacity, 1)
        self._n = 0
        self._timestamps = np.empty(self._capacity, dtype="datetime64[ns]")
        self._data = {f: np.empty(self._capacity, dtype=np.float64) for f in _FIELDS}

    def _grow(self) -> None:
        new_capacity = self._capacity * 2
        new_timestamps = np.empty(new_capacity, dtype="datetime64[ns]")
        new_timestamps[: self._n] = self._timestamps[: self._n]
        self._timestamps = new_timestamps
        for f in _FIELDS:
            new_arr = np.empty(new_capacity, dtype=np.float64)
            new_arr[: self._n] = self._data[f][: self._n]
            self._data[f] = new_arr
        self._capacity = new_capacity

    def append(self, bar: Bar) -> None:
        if self._n >= self._capacity:
            self._grow()

        i = self._n
        ts = bar.timestamp.replace(tzinfo=None) if bar.timestamp.tzinfo else bar.timestamp
        self._timestamps[i] = np.datetime64(ts)
        self._data["open"][i] = bar.open
        self._data["high"][i] = bar.high
        self._data["low"][i] = bar.low
        self._data["close"][i] = bar.close
        self._data["volume"][i] = bar.volume
        self._n += 1

    def __len__(self) -> int:
        return self._n

    def as_dataframe(self) -> pd.DataFrame:
        n = self._n
        return pd.DataFrame(
            {f: self._data[f][:n] for f in _FIELDS},
            index=pd.DatetimeIndex(self._timestamps[:n], name="timestamp"),
            copy=False,
        )
