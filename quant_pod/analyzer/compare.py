"""Side-by-side comparison table across multiple strategies/models."""
from __future__ import annotations

import pandas as pd

from quant_pod.analyzer import metrics as m


def compare(results: dict) -> pd.DataFrame:
    """`results` maps a name -> object with .equity_curve, .returns, .trades
    (i.e. a research.backtester.BacktestResult, or anything with those attrs).
    """
    rows = {}
    for name, res in results.items():
        rows[name] = m.summarize(res.equity_curve, res.returns, res.trades)
    return pd.DataFrame(rows).T
