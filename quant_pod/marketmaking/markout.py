"""Markout analysis: decompose each fill's eventual mark-to-market PnL into
spread capture (did we transact better than fair value?) vs. adverse selection
(did the price move against the new position afterward?) -- the standard
market-making diagnostic for telling "we're profitable because we're fairly
compensated for liquidity" apart from "we're profitable by luck" or
"we're being picked off by informed flow."

For a fill with `side_sign` (+1 bought, -1 sold) at `price`, against the mid
`mid_t` at the time of the fill and `mid_{t+N}` N steps later:

    total PnL per unit  = side_sign * (mid_{t+N} - price)
    spread capture      = side_sign * (mid_t      - price)   [priced better than fair value]
    adverse selection   = side_sign * (mid_{t+N}   - mid_t)   [subsequent drift for/against the position]

and total == spread capture + adverse selection by construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant_pod.marketmaking.simulator import MarketMakerResult


def compute_markouts(result: MarketMakerResult, horizons: list[int]) -> pd.DataFrame:
    """One row per fill. For each horizon N (in simulation steps), adds
    `spread_capture_N`, `adverse_selection_N`, and `total_markout_N` columns
    (in price units per unit traded); rows where step + N runs past the end of
    the simulation get NaN for that horizon.
    """
    rows = []
    n = len(result.mid) - 1
    for fill in result.fills:
        side_sign = 1.0 if fill.side == "buy" else -1.0
        mid_t = result.mid[fill.step]
        row = {
            "step": fill.step, "side": fill.side, "price": fill.price,
            "quantity": fill.quantity, "mid_at_fill": mid_t,
        }
        for horizon in horizons:
            later_step = fill.step + horizon
            if later_step > n:
                row[f"spread_capture_{horizon}"] = np.nan
                row[f"adverse_selection_{horizon}"] = np.nan
                row[f"total_markout_{horizon}"] = np.nan
                continue
            mid_later = result.mid[later_step]
            spread_capture = side_sign * (mid_t - fill.price)
            adverse_selection = side_sign * (mid_later - mid_t)
            row[f"spread_capture_{horizon}"] = spread_capture
            row[f"adverse_selection_{horizon}"] = adverse_selection
            row[f"total_markout_{horizon}"] = spread_capture + adverse_selection
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_markouts(markouts: pd.DataFrame, horizons: list[int]) -> dict:
    """Average per-unit spread capture / adverse selection / total at each
    horizon, plus the fill count each average is over (NaN-dropping, since
    fills near the end of the simulation lack a full-horizon markout).
    """
    stats = {}
    for horizon in horizons:
        spread_col = markouts[f"spread_capture_{horizon}"]
        adverse_col = markouts[f"adverse_selection_{horizon}"]
        total_col = markouts[f"total_markout_{horizon}"]
        stats[horizon] = {
            "avg_spread_capture": float(spread_col.mean()),
            "avg_adverse_selection": float(adverse_col.mean()),
            "avg_total_markout": float(total_col.mean()),
            "n": int(total_col.notna().sum()),
        }
    return stats


def format_markout_table(markout_stats: dict, horizons: list[int]) -> str:
    """Shared formatter for the two CLI entry points that print a markout
    decomposition (`scripts/run_market_maker.py` and `trader.cli quote`), so
    the table layout and `summarize_markouts`' dict schema only need to agree
    in one place.
    """
    lines = [f"{'horizon':>8s} {'n':>6s} {'spread capture':>16s} {'adverse selection':>18s} {'total':>10s}"]
    for h in horizons:
        s = markout_stats[h]
        lines.append(
            f"{h:>8d} {s['n']:>6d} {s['avg_spread_capture']:>16.4f} "
            f"{s['avg_adverse_selection']:>18.4f} {s['avg_total_markout']:>10.4f}"
        )
    return "\n".join(lines)
