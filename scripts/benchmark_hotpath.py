"""Isolated microbenchmark of the matching-engine hot path: pure Python
arithmetic vs. the C++ pybind11 call, outside the asyncio/zmq event loop so
scheduler jitter doesn't pollute the percentiles.

Usage: python scripts/benchmark_hotpath.py [n_iterations]
"""
from __future__ import annotations

import sys
import time

import numpy as np

from quant_pod.trader.hotpath import OrderBook, synthetic_fill_price
from quant_pod.trader.py_orderbook import PyOrderBook


def python_fill_price(is_buy: bool, reference_price: float, slippage_bps: float) -> float:
    direction = 1.0 if is_buy else -1.0
    return reference_price * (1 + direction * slippage_bps / 10_000)


def bench(fn, n: int) -> np.ndarray:
    samples = np.empty(n, dtype=np.float64)
    is_buy = True
    for i in range(n):
        t0 = time.perf_counter_ns()
        fn(is_buy, 150.0, 1.0)
        t1 = time.perf_counter_ns()
        samples[i] = t1 - t0
        is_buy = not is_buy
    return samples


def bench_deep_book_match(book_cls, n: int, n_levels: int, qty_per_level: float, incoming_qty: float) -> np.ndarray:
    """Each trial: build a fresh n_levels-deep ask book, then time matching one
    incoming buy order that walks through most of those levels. Building the
    book is excluded from the timed region -- only the match call is timed.
    """
    samples = np.empty(n, dtype=np.float64)
    for i in range(n):
        book = book_cls()
        for level in range(n_levels):
            book.add_limit_order(level, False, 100.0 + level * 0.01, qty_per_level)
        t0 = time.perf_counter_ns()
        book.add_market_order(999_999, True, incoming_qty)
        t1 = time.perf_counter_ns()
        samples[i] = t1 - t0
    return samples


def report(label: str, samples: np.ndarray) -> None:
    print(
        f"{label:28s} n={len(samples):>7d} "
        f"mean={samples.mean():8.1f}ns p50={np.percentile(samples, 50):8.1f}ns "
        f"p95={np.percentile(samples, 95):8.1f}ns p99={np.percentile(samples, 99):8.1f}ns"
    )


def speedup_line(py_samples: np.ndarray, cpp_samples: np.ndarray) -> str:
    speedup = py_samples.mean() / cpp_samples.mean()
    return f"C++ mean is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than pure Python."


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000

    print("=== Single trivial fill-price calculation (one FFI call vs. one Python expression) ===")
    bench(python_fill_price, 1_000)  # warm up
    bench(synthetic_fill_price, 1_000)
    py_trivial = bench(python_fill_price, n)
    cpp_trivial = bench(synthetic_fill_price, n)
    report("Python arithmetic", py_trivial)
    report("C++ hot path (pybind11)", cpp_trivial)
    print(speedup_line(py_trivial, cpp_trivial))

    print("\n=== Matching against a 50-level-deep order book (real per-call work) ===")
    n_deep = max(n // 20, 2_000)  # building a book each trial is much heavier than the trivial case
    bench_deep_book_match(PyOrderBook, 100, 50, 100.0, 4_000.0)  # warm up
    bench_deep_book_match(OrderBook, 100, 50, 100.0, 4_000.0)
    py_deep = bench_deep_book_match(PyOrderBook, n_deep, 50, 100.0, 4_000.0)
    cpp_deep = bench_deep_book_match(OrderBook, n_deep, 50, 100.0, 4_000.0)
    report("Python OrderBook", py_deep)
    report("C++ OrderBook", cpp_deep)
    print(speedup_line(py_deep, cpp_deep))
