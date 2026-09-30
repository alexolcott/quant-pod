"""Common interface every data source adapter must implement.

New sources (a different broker's historical API, a CSV dump, etc.) just need
to implement `fetch` and can be dropped into the CLI without touching the
store or replay code.
"""
from __future__ import annotations

from datetime import datetime
from typing import Protocol

import pandas as pd


class DataSource(Protocol):
    def fetch(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Return a DataFrame indexed by UTC timestamp with columns:
        open, high, low, close, volume.
        """
        ...
