"""Isolated benchmark: the old per-tick `pd.DataFrame(list_of_dicts)` rebuild
vs. BarBuffer's pre-allocated-array append, across a run of n_bars ticks.

Usage: python scripts/benchmark_bar_buffer.py [n_bars]
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from quant_pod.common.events import Bar
from quant_pod.trader.bar_buffer import BarBuffer


def make_bars(n: int) -> list[Bar]:
    start = datetime(2020, 1, 1)
    return [
        Bar(symbol="AAPL", timestamp=start + timedelta(days=i), open=100, high=101, low=99, close=100.5, volume=1_000)
        for i in range(n)
    ]


def bench_old_approach(bars: list[Bar]) -> np.ndarray:
    samples = np.empty(len(bars), dtype=np.float64)
    bar_rows: list[dict] = []
    for i, bar in enumerate(bars):
        bar_rows.append(
            {"timestamp": bar.timestamp, "open": bar.open, "high": bar.high,
             "low": bar.low, "close": bar.close, "volume": bar.volume}
        )
        t0 = time.perf_counter_ns()
        pd.DataFrame(bar_rows).set_index("timestamp")
        t1 = time.perf_counter_ns()
        samples[i] = t1 - t0
    return samples


def bench_bar_buffer(bars: list[Bar]) -> np.ndarray:
    samples = np.empty(len(bars), dtype=np.float64)
    buf = BarBuffer()
    for i, bar in enumerate(bars):
        buf.append(bar)
        t0 = time.perf_counter_ns()
        buf.as_dataframe()
        t1 = time.perf_counter_ns()
        samples[i] = t1 - t0
    return samples


def report(label: str, samples: np.ndarray) -> None:
    print(
        f"{label:24s} n={len(samples):>6d} "
        f"mean={samples.mean() / 1000:8.2f}us p50={np.percentile(samples, 50) / 1000:8.2f}us "
        f"p95={np.percentile(samples, 95) / 1000:8.2f}us "
        f"last-tick={samples[-1] / 1000:8.2f}us "
        f"total={samples.sum() / 1_000_000:8.2f}ms"
    )


if __name__ == "__main__":
    n_bars = int(sys.argv[1]) if len(sys.argv) > 1 else 1258  # matches the AAPL 5yr dataset used elsewhere
    bars = make_bars(n_bars)

    old_samples = bench_old_approach(bars)
    new_samples = bench_bar_buffer(bars)

    print(f"Simulating a {n_bars}-bar live run, timing just the per-tick DataFrame construction:\n")
    report("list[dict] -> DataFrame", old_samples)
    report("BarBuffer", new_samples)

    print(f"\nLast-tick speedup: {old_samples[-1] / new_samples[-1]:.1f}x")
    print(f"Total-time-over-the-run speedup: {old_samples.sum() / new_samples.sum():.1f}x")
