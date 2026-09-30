# quant-pod guide

This is the deep-dive companion to the top-level `README.md`. The README gets
you running in five minutes; this explains what every piece actually does,
why it's built the way it is, and how to extend it.

## Contents

- [System overview](#system-overview)
- [`common/` — shared schemas](#common--shared-schemas)
- [`ingester/` — data in](#ingester--data-in)
- [`research/` — strategies and backtesting](#research--strategies-and-backtesting)
- [`analyzer/` — performance metrics and reports](#analyzer--performance-metrics-and-reports)
- [`trader/` — the live engine](#trader--the-live-engine)
- [`trader/hotpath/` — the C++ matching engine](#traderhotpath--the-c-matching-engine)
- [`dashboard/` — the Streamlit UI](#dashboard--the-streamlit-ui)
- [Full command reference](#full-command-reference)
- [Common workflows](#common-workflows)
- [Extending the pod](#extending-the-pod)

---

## System overview

```
ingester (fetch + store)  ---->  research (backtest strategies)
        |
        v
ingester.replay (ZeroMQ PUB, its own process)
        |
        v
trader.engine (ZeroMQ SUB, its own process) --orders--> Python or C++ matching engine
        |
        v
analyzer (same metrics/report code the backtester uses)
```

The design principle that ties this together: **a `Strategy` subclass is
written once** (`research/strategy.py`) and runs unmodified in two very
different contexts — the vectorized backtester (fast, for research) and the
live event-driven trader (bar-by-bar, for the "real" run). Because both paths
call `strategy.generate_target_positions(...)` / `strategy.target_position_for(...)`,
a backtest result and a live-replay result for the same strategy are directly
comparable — there's no separate "live version" of the logic that could drift
out of sync with what you researched.

Each stage also runs as its own OS process when it matters (`ingester.replay`
and `trader.engine` communicate over a ZeroMQ socket, not a shared Python
object) — that's deliberate: it's what makes this a simulated *pod* of
services rather than one monolithic script, and it's what makes the tick-to-fill
latency numbers the trader reports meaningful (they include real inter-process
message delivery, not a function call).

---

## `common/` — shared schemas

Everything else imports from here. Three files:

- **`events.py`** — the Pydantic models every module passes around:
  - `Bar` — one OHLCV candle for a symbol. What the ingester stores and replays.
  - `Order` — a request to trade (`symbol`, `side`, `quantity`, `order_type`). Stamps
    itself with `created_ns` (a nanosecond timestamp) the moment it's constructed —
    this is the start of the latency measurement the trader reports.
  - `Fill` — the result of an order being matched, with its own `filled_ns` timestamp.
  - `Signal` — a strategy's target position for a symbol (not heavily used yet, but
    there if you want to decouple signal generation from order sizing later).
- **`config.py`** — `DATA_DIR`, `REPORTS_DIR`, and `REPLAY_ZMQ_ADDR` (the socket
  address the replay publisher binds and the trader connects to:
  `tcp://127.0.0.1:5556`, local-machine only).
- **`logging.py`** — one `get_logger(name)` used by every CLI/process so log
  lines look the same everywhere (`HH:MM:SS [LEVEL] name: message`).

---

## `ingester/` — data in

**Job:** get historical price data from *somewhere*, normalize it to one
schema, store it, and be able to replay it later as if it were live.

- **`sources/`** — pluggable adapters, each implementing one method:
  `fetch(symbol, start, end) -> DataFrame` with columns `open/high/low/close/volume`.
  - `yfinance_source.py` — real data from Yahoo Finance, no API key.
  - `synthetic_source.py` — generates a GBM (geometric Brownian motion) price
    path, seeded deterministically by symbol name so the same ticker always
    produces the same fake data. Useful for testing offline or when Yahoo
    Finance is rate-limiting you.
  - Adding a new source (e.g. a different free API, a CSV dump) means writing
    one class with a `fetch` method — nothing else needs to change.
- **`store.py`** — reads/writes Parquet files in `data/<SYMBOL>.parquet`.
  `write_bars` merges new data into what's already there and de-dupes on
  timestamp (last write wins), so re-fetching an overlapping date range is safe.
- **`cli.py`** — the `fetch` and `list` commands (see [command reference](#full-command-reference)).
- **`replay.py`** — the piece that turns stored history into a *live feed*. It
  reads a symbol's bars and publishes them one at a time over a ZeroMQ PUB
  socket, at a configurable pace (`--speed`, in seconds between bars; `0` =
  as fast as possible). Sends a sentinel message (`__END__`) when done so
  subscribers know to stop. Runs as its own process — start the trader
  *first* so it's already subscribed before the publisher sends anything
  (ZeroMQ PUB doesn't buffer for subscribers that haven't connected yet —
  the "slow joiner" problem — which is why `replay.py` also pauses briefly
  after binding before it starts sending).

---

## `research/` — strategies and backtesting

**Job:** let you develop and evaluate a trading idea quickly, without needing
a live feed running.

- **`strategy.py`** — the `Strategy` abstract base class. You implement one
  method, `generate_target_positions(bars) -> Series`, which maps a history
  of bars to a target share position at each timestamp. `target_position_for(...)`
  is a convenience the live trader uses to ask "given everything up to now,
  what should my position be?"
- **`strategies/moving_average.py`** — `MovingAverageCrossover`: long a fixed
  number of shares when a fast moving average is above a slow one, flat
  otherwise. The simplest possible trend-following strategy.
- **`strategies/risk_ratio.py`** — `RiskRatioStrategy`: a reconstruction of
  the composite Sharpe/VaR/drawdown ratio idea from your standalone Risk
  Metric project — rolling Sharpe, historical VaR, and drawdown are each
  z-scored and summed into one number; buys when that composite score drops
  below a threshold (i.e. the stock looks oversold relative to its own recent
  risk profile).
- **`backtester.py`** — `run_backtest(strategy, bars)`. Vectorized (pandas,
  not a bar-by-bar loop) for speed. Important detail: it trades on the
  **next** bar's open, never the bar the signal was computed on — a signal
  computed from bar N's close can't be filled at bar N's own open, since
  that would be trading on information from the future. This is checked
  directly in `tests/test_backtester.py::test_backtest_has_no_lookahead_bias`.
  Returns a `BacktestResult` with `equity_curve`, `positions`, `trades`, `returns`.

---

## `analyzer/` — performance metrics and reports

**Job:** turn an equity curve + trade log into the numbers people actually
look at, the same way whether it came from a backtest or a live run.

- **`metrics.py`** — the formulas:
  - `cagr` — annualized growth rate.
  - `sharpe_ratio` / `sortino_ratio` — risk-adjusted return (Sortino only
    penalizes downside volatility).
  - `max_drawdown` — worst peak-to-trough decline.
  - `calmar_ratio` — CAGR divided by max drawdown (return per unit of worst pain).
  - `historical_var` — historical Value-at-Risk (the loss you'd expect to
    exceed only 5% of the time, by default).
  - `turnover` — gross traded value relative to average equity (how much
    the strategy churns).
  - `summarize(...)` bundles all of the above into one dict.
- **`report.py`** — `generate_report(name, equity_curve, returns, trades)`
  prints a formatted text summary, saves a matplotlib equity-curve PNG to
  `reports/`, and returns everything as a dict too.
- **`compare.py`** — `compare({name: result, ...})` builds a side-by-side
  DataFrame of every strategy's metrics, for exactly the kind of "which one's
  actually better" question you'd ask after researching a few ideas.

---

## `trader/` — the live engine

**Job:** actually run a strategy against the live-replayed feed, in real
time, with the kind of machinery a real trading system has: risk checks,
a matching engine, and latency instrumentation.

- **`engine.py`** — `run_trader(symbol, strategy, engine, ...)`, an asyncio
  loop:
  1. Subscribe to the replay feed over ZeroMQ.
  2. On each new bar: append it to a running history, ask the strategy for
     a target position, diff against the current position to get an order
     quantity.
  3. If nonzero, build an `Order`, run it through `risk_limits.check(...)`,
     and if it passes, send it to the matching engine (`sim_exchange.PySimExchange`
     or `hotpath_exchange.CppSimExchange`, selected by `--engine`).
  4. Record two latency numbers per order: **decision latency** (tick
     arrival → order construction, i.e. how long the strategy + order-building
     code took) and **matching latency** (order → fill, i.e. how long the
     matching engine itself took — this is the number that differs between
     the Python and C++ backends).
  5. Mark equity to market on every bar (`cash + position * close`), building
     an equity curve exactly like the backtester's, so `analyzer` can report
     on it identically.
  6. Stop on the replay's `__END__` sentinel.
- **`sim_exchange.py`** — `PySimExchange`: the plain-Python matching backend.
  For a market order, just applies a slippage-adjusted price to the current
  reference price. Intentionally trivial — it's the baseline the C++ path is
  benchmarked against.
- **`hotpath_exchange.py`** — `CppSimExchange`: same interface, but the price
  calculation itself happens in the compiled `hotpath` extension. See the
  next section for why it's built the way it is.
- **`risk_limits.py`** — `RiskLimits.check(order, current_position, reference_price)`:
  two pre-trade checks (max absolute position, max single-order notional).
  Runs on every order, so it's kept cheap on purpose.
- **`latency.py`** — `LatencyTracker`: accumulates nanosecond samples and
  reports mean/p50/p95/p99/max in microseconds.
- **`cli.py`** — the `trade` command (see [command reference](#full-command-reference)).

---

## `trader/hotpath/` — the C++ matching engine

This is a [pybind11](https://pybind11.readthedocs.io/) extension
(`quant_pod/trader/hotpath/`, compiled to `_hotpath*.pyd`/`.so`) with two
things in it, and the reason there are two is itself the interesting part.

**`OrderBook`** (`orderbook.hpp`/`.cpp`) is a real price-time-priority limit
order book: bids and asks are kept in `std::map`s (highest-bid-first,
lowest-ask-first), each price level is a FIFO queue of resting orders,
`add_limit_order` matches against the opposite side and rests any
unfilled remainder, `add_market_order` matches immediate-or-cancel (drops
any unfilled remainder rather than resting it). It's tested directly in
`tests/test_orderbook.py`, in parallel with a pure-Python reference
implementation (`trader/py_orderbook.py`) that implements the identical
interface — every test runs against both, so they're guaranteed to agree.

**`synthetic_fill_price`** is what the live trader actually calls per order.
Here's why it's separate from `OrderBook`: the first version of
`CppSimExchange` routed every order through two `OrderBook` calls (seed a
synthetic resting order on the opposite side, then match against it) —
correct, but it crosses the Python/C++ boundary twice and pays for a
`std::map` insert/erase and a heap-allocated `std::vector<BookFill>` *per
order*. Benchmarked, that came in around **13µs per order — slower than the
~1.6µs pure-Python version it was supposed to beat.** `synthetic_fill_price`
is what replaced it: one call, no map operations, no heap allocation, just
returns a `double`.

The honest result, from `scripts/benchmark_hotpath.py` (see
[command reference](#full-command-reference) to run it yourself):

| Workload | Python | C++ | |
|---|---|---|---|
| One trivial value (`synthetic_fill_price`) | ~95ns | ~155ns | **Python wins** — FFI call overhead costs more than the float multiply it avoids |
| Matching against a 50-level-deep order book | ~16.4µs | ~8.5µs | **C++ wins, ~2x** — real per-call work amortizes the FFI crossing cost |

The lesson (and the reason both paths are kept in the codebase rather than
just shipping the fast one): a compiled extension only pays for itself once
there's enough work per call to amortize the cost of crossing into it. For a
single arithmetic operation, it doesn't. For real order-matching work, it
does.

---

## `dashboard/` — the Streamlit UI

`dashboard/app.py` is an interactive viewer, not a separate implementation —
every number it shows comes from calling `research.backtester.run_backtest`
and `analyzer.metrics.summarize` directly, so it can never disagree with the
CLI.

- **Sidebar** — pick a cached symbol or fetch a new one (calls
  `YFinanceSource` + `store.write_bars`, same as the CLI); pick a strategy
  with live parameter sliders (fast/slow window, position size, etc.).
- **Single-strategy view** — metric tiles, an interactive Plotly equity
  curve, a shaded drawdown chart underneath it, and the full trade log table.
- **Comparison view** ("Compare all strategies" checkbox) — runs every
  strategy with its default parameters and shows a side-by-side metrics
  table plus overlaid equity curves with a legend.

It only covers backtests, not the live trader — the live trader is a
separate asyncio process talking over a network socket, which doesn't fit
naturally into Streamlit's request/response model, so it stays a CLI/script
thing for now.

---

## Full command reference

All commands assume the venv is active (`.venv\Scripts\activate`) and you're
in the `quant-pod` directory.

### Ingester

```
python -m quant_pod.ingester.cli fetch <SYMBOL> --start YYYY-MM-DD --end YYYY-MM-DD [--source yfinance|synthetic]
python -m quant_pod.ingester.cli list
```

### Replay (run before or after the trader — see workflows below)

```
python -m quant_pod.ingester.replay <SYMBOL> [--speed 0.0] [--warmup 1.0]
```
`--speed` is seconds between bars (`0` = as fast as possible). `--warmup` is
how long to pause after binding before sending, to give the trader time to connect.

### Trader

```
python -m quant_pod.trader.cli <SYMBOL> [--strategy moving_average|risk_ratio] [--engine python|cpp] [--report/--no-report]
```

### Orchestrator (does ingest-if-needed + replay + trade + report in one command)

```
python scripts/run_pod.py <SYMBOL> [--strategy ...] [--engine ...] [--years 5] [--replay-speed 0.0]
```

### Benchmark

```
python scripts/benchmark_hotpath.py [n_iterations]
```

### Dashboard

```
streamlit run quant_pod/dashboard/app.py
```

### Tests

```
pytest
pytest tests/test_orderbook.py -v   # just one file
```

---

## Common workflows

**"I have a new strategy idea and want to see if it works."**
Skip the live trader entirely — just use the backtester (fast, no
subprocess juggling):
```python
from quant_pod.ingester import store
from quant_pod.research.backtester import run_backtest
from quant_pod.analyzer.report import generate_report
from quant_pod.research.strategies.moving_average import MovingAverageCrossover

bars = store.read_bars("AAPL")
result = run_backtest(MovingAverageCrossover(fast=10, slow=30), bars)
generate_report("AAPL fast MA", result.equity_curve, result.returns, result.trades)
```
Or do the same thing interactively in the dashboard.

**"I want to see it run 'live' end to end."**
```
python scripts/run_pod.py AAPL --strategy moving_average --engine cpp
```
This is the one-command version of: fetch data if missing, start the
trader, wait for it to connect, start the replay publisher, wait for both
to finish, print the report.

**"I want to run it manually, in separate terminals, to watch the pieces talk to each other."**
Terminal 1 (start this first — it needs to be connected before the publisher sends):
```
python -m quant_pod.trader.cli AAPL --strategy moving_average --engine python
```
Terminal 2 (after terminal 1 says "Trader started"):
```
python -m quant_pod.ingester.replay AAPL --speed 0.05
```
`--speed 0.05` slows the replay down (50ms between bars) so you can actually
watch it happen instead of it finishing in under a second.

**"I want to know if the C++ path is actually worth it here."**
```
python scripts/benchmark_hotpath.py 200000
```
Read the table in [the hotpath section above](#traderhotpath--the-c-matching-engine)
for what the numbers mean.

---

## Extending the pod

**Add a new strategy.** Create a class in `quant_pod/research/strategies/`
implementing `Strategy.generate_target_positions(bars) -> pd.Series`. Register
it in `STRATEGIES` in both `quant_pod/trader/cli.py` and
`quant_pod/dashboard/app.py` to make it available from the CLI and dashboard.
No changes needed anywhere else — the backtester and live trader both take
any `Strategy` instance.

**Add a new data source.** Create a class in `quant_pod/ingester/sources/`
implementing `fetch(symbol, start, end) -> pd.DataFrame` (columns
`open/high/low/close/volume`, indexed by UTC timestamp). Register it in the
`SOURCES` dict in `quant_pod/ingester/cli.py`.

**Add a new matching backend.** Implement a class with
`match(order: Order, reference_price: float) -> Fill` (see
`sim_exchange.PySimExchange` for the minimal shape) and register it in
`_make_exchange(...)` in `quant_pod/trader/engine.py`.

**Trade multiple symbols at once.** Right now `replay.py` and `engine.py`
are both single-symbol. The ZeroMQ topic is already the symbol name
(`sock.send_multipart([topic, payload])`), so multi-symbol support mostly
means running one replay publisher per symbol (or one publisher multiplexing
several) and having the trader subscribe to more than one topic and track
per-symbol position/equity separately.
