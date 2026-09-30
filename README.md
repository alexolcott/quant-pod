# quant-pod

A simulated quant trading pod: a standardized data **ingester**, a strategy
**researcher**, a performance **analyzer**, and a low-latency-simulated
**trader**, wired together the way a real pod's services would be.

See **[docs/GUIDE.md](docs/GUIDE.md)** for a full walkthrough of every
module, a complete command reference, and common workflows. This README is
just the quickstart.

```
ingester (fetch + store)  ->  research (backtest strategies)
        |
        v
ingester.replay (ZeroMQ PUB, separate process)
        |
        v
trader.engine (ZeroMQ SUB, separate process) --matches orders via--> Python or C++ hot path
        |
        v
analyzer (same metrics/report code path as backtests)
```

The same `Strategy` class runs unmodified in both the vectorized backtester
(research) and the live event-driven trader, so backtest and "live" results
are directly comparable.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install setuptools wheel          # needed once, before an editable install with --no-build-isolation
pip install --no-build-isolation -e ".[dev]"
```

Building the C++ extension requires a compiler; on Windows this repo was
built against Visual Studio Build Tools, which `setuptools` finds
automatically (no need to open a Developer Command Prompt).

## Quickstart

```
# one command: ingest (if needed) -> replay -> trade -> report
python scripts/run_pod.py AAPL --strategy moving_average --engine cpp
```

Or run each piece by hand, in separate terminals (the trader should be
started first so it's connected before the replay publisher starts sending):

```
python -m quant_pod.ingester.cli fetch AAPL --start 2019-01-01 --end 2024-01-01
python -m quant_pod.trader.cli AAPL --strategy moving_average --engine python
python -m quant_pod.ingester.replay AAPL --speed 0
```

## Research (fast iteration, no live feed needed)

```python
from quant_pod.ingester import store
from quant_pod.research.strategies.moving_average import MovingAverageCrossover
from quant_pod.research.backtester import run_backtest
from quant_pod.analyzer.report import generate_report

bars = store.read_bars("AAPL")
result = run_backtest(MovingAverageCrossover(fast=20, slow=50), bars)
generate_report("AAPL moving_average backtest", result.equity_curve, result.returns, result.trades)
```

## Dashboard

An interactive Streamlit view over the research pipeline: pick a symbol and
strategy (or fetch a new symbol), run a backtest, and see the equity curve,
drawdown, metrics, and trade log update live. Toggle "Compare all strategies"
for a side-by-side table and overlaid equity curves.

```
pip install -e ".[dashboard]"
streamlit run quant_pod/dashboard/app.py
```

It's a thin viewer on top of `research.backtester` and `analyzer.metrics` —
no separate logic of its own — so anything it shows matches the CLI/report
output exactly.

## The C++ hot path: what actually got faster, and what didn't

`quant_pod/trader/hotpath/` is a pybind11 extension with two things in it:

- `synthetic_fill_price(...)`: one trivial arithmetic calculation, used for
  the live trader's per-order pricing.
- `OrderBook`: a real price-time-priority limit order book (std::map + FIFO
  queues per price level), independently tested in `tests/test_orderbook.py`
  against a pure-Python reference implementation (`trader/py_orderbook.py`).

`scripts/benchmark_hotpath.py` benchmarks both, isolated from asyncio/zmq
scheduling jitter, and the result is genuinely two-sided:

- For **one trivial value per call**, pure Python (~95ns) beats the C++
  pybind11 call (~155ns) — the cost of crossing the FFI boundary exceeds the
  cost of the float multiply it's avoiding. An earlier version routed every
  order through two `OrderBook` calls (seed synthetic liquidity, then match
  it) instead; that paid for a `std::map` insert/erase and a
  `std::vector<BookFill>` allocation per order and came in at ~13µs — *slower*
  than pure Python. That's why the live trader's hot path uses
  `synthetic_fill_price` instead.
- For **matching against a 50-level-deep order book** (real, non-trivial
  per-call work), the C++ `OrderBook` is ~2x faster than the Python
  reference implementation — this is where compiled code actually pays for
  itself.

Run it yourself:

```
python scripts/benchmark_hotpath.py 200000
```

## Layout

- `quant_pod/common/` — shared event schemas (`Bar`, `Order`, `Fill`, `Signal`), config, logging.
- `quant_pod/ingester/` — pluggable data sources (`yfinance`, `synthetic`), Parquet store, CLI, and the ZeroMQ replay publisher.
- `quant_pod/research/` — `Strategy` base class, example strategies, vectorized backtester.
- `quant_pod/analyzer/` — Sharpe/Sortino/CAGR/MaxDD/Calmar/VaR/turnover, report generation, multi-strategy comparison.
- `quant_pod/trader/` — the live event-driven engine, risk limits, latency tracking, and the C++/Python matching backends.
- `quant_pod/dashboard/` — Streamlit research dashboard (optional, `pip install -e ".[dashboard]"`).
- `scripts/run_pod.py` — orchestrates the whole pod in one command.
- `scripts/benchmark_hotpath.py` — isolated Python-vs-C++ latency benchmark.

## Tests

```
pytest
```
