# The Avellaneda-Stoikov market maker: assumptions, calibration, and results

A one-pager for defending this piece of quant-pod in an interview: what it
assumes, what the statistics actually show, and what's deliberately
simplified. See `docs/GUIDE.md` for the rest of the pod; this covers only
`quant_pod/marketmaking/`, `quant_pod/analyzer/robust_stats.py`, and
`quant_pod/trader/market_maker_engine.py`.

## What it is

An implementation of Avellaneda & Stoikov's 2008 optimal market-making model
(`quant_pod/marketmaking/avellaneda_stoikov.py`): quote a reservation price
that skews away from mid as inventory builds up, and an optimal total spread
around it, both derived from a stochastic-control argument over a CARA
(exponential) utility function. It runs in two places with the same quoting
and fill-probability code: a synthetic research simulator
(`simulator.py`, driven by a simulated Brownian mid-price path) and a live
loop (`trader/market_maker_engine.py`, driven by the pod's real ZeroMQ bar
feed) — the same "same code, backtest and live" story the rest of the pod
tells for bar-based strategies, extended to a two-sided quoter.

## Model and what it assumes

- **Reservation price** `r = mid - q·γ·σ²·(T-t)`: the price at which the
  maker is indifferent to holding inventory `q`. Long inventory skews `r`
  below mid (eager to sell); short skews it above (eager to buy). This term
  is *linear* in `γ` for fixed `q`, so skew magnitude always grows with risk
  aversion — one of the few things about this model that's unconditionally
  true.
- **Optimal spread** `γσ²(T-t) + (2/γ)·ln(1+γ/κ)`: **not** monotonic in
  `γ`. The first (inventory-risk) term grows with risk aversion; the second
  (order-arrival) term *shrinks* as `γ` grows. Which one dominates depends on
  `σ²(T-t)` relative to `κ` — this surprised me when a test assuming
  monotonicity failed, and it's a real property of the formula, not a bug
  (see the docstring in `avellaneda_stoikov.py` and the two
  `test_spread_widens...` tests in `tests/test_avellaneda_stoikov.py` for the
  regime where it does and doesn't hold).
- **Fill model**: a counterparty hits a quote `δ` away from mid with
  intensity `λ(δ) = A·e^(−κδ)` (Poisson). Fills are drawn directly from this
  intensity, not matched through `trader.hotpath.OrderBook` — that engine has
  no cancellation, and this strategy replaces both quotes every tick, so a
  persistent book would just accumulate stale resting orders forever. This is
  the standard way the AS model family is simulated in the literature, not a
  shortcut specific to this project.

## Calibration

`γ`, `κ`, `σ`, and `A` are set to illustrative values in the same ballpark as
the values the original paper and common tutorials use, scaled to the
instrument — **not fit to real market data**. That's an honest limitation: a
real desk calibrates `κ` from an empirical fill-intensity-vs-distance curve
measured against live quotes, and `σ` from a short realized-vol estimator,
not assumed. In the live engine (`market_maker_engine.py`), `σ` *is*
estimated on the fly from a rolling window of real bar closes; `κ` and `A`
are still config knobs, not fit.

`time_horizon` in the live engine is a constant, refreshed every tick,
rather than a single terminal time — a live feed doesn't know in advance how
many bars remain. In effect: "size for unwinding this inventory over roughly
this many bars," not a single end-of-day liquidation. This is a deliberate
simplification of the paper's finite-horizon setup, not an oversight.

## What the backtest statistics actually show

Run against this repo's real cached S&P 500 data (`scripts/backtest_universe.py`,
applies to the pre-existing bar strategies, not the market maker, but it's
the clearest real evidence for the method):

- `moving_average`'s best performer by raw Sharpe across 501 symbols was
  SNDK (Sharpe 1.62). Its **deflated Sharpe ratio is 15.6%** — after
  accounting for having tried 501 symbols, there's only a 15.6% probability
  this reflects real skill rather than the best of 501 noisy draws.
- `risk_ratio`'s best performer, PG (Sharpe 1.60), deflates to **69.6%** —
  a much more defensible pick on the same raw Sharpe.
- A separate check (`scripts/walk_forward_validate.py`) on SNDK specifically
  — purged walk-forward CV, refitting `fast`/`slow` on train data only —
  found **no** evidence of parameter overfitting (out-of-sample Sharpe
  slightly *beat* the naive whole-history-optimized Sharpe). That's not a
  contradiction: DSR catches **cross-symbol selection bias** (501 tickers
  tried), walk-forward CV catches **within-symbol parameter overfitting** (9
  `fast`/`slow` combinations tried on one ticker). SNDK fails the first check
  and passes the second — they're different failure modes, and a strategy
  can fail one without the other.

For the market maker itself, a live run against real AAPL daily bars
(`python -m quant_pod.trader.cli quote AAPL`) over 1,946 bars produced 856
fills and a markout decomposition where **spread capture (~0.68/unit)
dominates and adverse selection is small and sign-mixed across horizons**
(see Limitations — this is expected given the current flow model, not
evidence of a particularly good strategy).

## Known limitations (say these before an interviewer finds them)

- **No informed/toxic flow component.** The synthetic order flow (and the
  live engine's fill draws) are independent of future price moves, so
  adverse selection should average to ~0 by construction — and it does. The
  markout diagnostic is working correctly; it just isn't being given a flow
  with real information content to detect. Adding a toxic-flow component
  (fills correlated with the next price innovation) is the natural next step
  to make this diagnostic — and the risk controls meant to respond to it —
  actually earn their keep.
- **No real limit order book with competing liquidity.** Fills are
  probabilistic, not matched against other participants' resting orders (see
  Model, above). `trader.hotpath.OrderBook` would need a cancellation API
  before it could support a persistent, continuously-requoted book.
- **Parameters aren't fit to data** (see Calibration).
- **The live engine runs on daily bars**, not tick/quote data — "live" here
  means "driven by the pod's real streaming infrastructure," not
  "sub-second market making." `arrival_rate` is calibrated per-bar, not
  per-second, and isn't comparable across the two engines without rescaling.
- **Terminal inventory is marked-to-market, not force-liquidated** in the
  research simulator.

## Likely interview follow-ups

- *"Walk me through why the reservation price moves away from mid."* —
  CARA-utility argument: a risk-averse maker holding inventory `q` values an
  additional unit at the certainty-equivalent price, which embeds the cost of
  the inventory's variance over the remaining horizon; that cost is linear in
  `q·γ·σ²·(T-t)`.
- *"Why doesn't the spread just widen with risk aversion?"* — see the
  non-monotonicity note above; be ready to state which term dominates in
  which regime.
- *"How do you know this isn't overfit?"* — this is the DSR/walk-forward
  answer above, with real numbers, not "I ran one backtest and the Sharpe
  looked good."
- *"What's missing for this to be real?"* — the Limitations section, in that
  order of importance: toxic flow, a real matching book, fit parameters,
  tick data.
