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
- **Quotes never cross mid** (`avellaneda_stoikov.suppress_quotes_crossing_mid`):
  the inventory skew `q·γ·σ²·(T-t)` can, for a large enough position, sigma,
  or `time_horizon`, exceed half the spread — pushing the *ask* below the
  current mid (or the bid above it). Found this by tracing a real bug: buying
  10 AAPL shares at $40 skewed the reservation price enough that the ask
  dropped to $29 one tick later, selling that inventory at a guaranteed
  ~$11/share loss relative to fair value — not inventory risk, just quoting a
  self-destructive price. When skew would push a side past mid, that side is
  suppressed (NaN) instead, the same response `max_inventory` uses for its
  own limit. This is a guard on top of the textbook formula, not part of the
  original paper — the paper doesn't address what happens when its own
  closed-form skew term overshoots the spread.

## Calibration

`γ`, `κ`, `σ`, and `A` are set to illustrative values in the same ballpark as
the values the original paper and common tutorials use, scaled to the
instrument — **not fit to real market data** by default. That's an honest
limitation: a real desk calibrates `κ` from an empirical fill-intensity-vs-distance
curve measured against live quotes, and `σ` from a short realized-vol
estimator, not assumed. In the live engine (`market_maker_engine.py`), `σ`
*is* estimated on the fly from a rolling window of real bar closes; `κ` and
`A` are still config knobs there, not fit.

`quant_pod/marketmaking/calibration.py` does real calibration work, within
what daily OHLCV bars actually make possible:

- **`σ` — genuinely calibrated.** `garman_klass_sigma` uses the full
  high/low/open/close range (Garman & Klass, 1980), a real, textbook
  improvement over a naive close-to-close diff for the same window — and,
  computed in log-return space and scaled back to dollars at the current
  price, it fixes the exact price-level bug found earlier (AAPL's dollar vol
  looked smaller in 2022 than 2019 purely because the price had tripled).
- **`A` (arrival_rate) — tied to real volume, not independently calibrated.**
  `arrival_rate_from_volume` scales average traded volume by an assumed
  participation rate and `fill_size`. The participation rate is a judgment
  call about market share, not a fit.
- **`κ` — not calibrated, by necessity.** A textbook calibration needs
  quote/fill-level data (who got hit at what distance from the touch) that
  daily bars don't have. `calibrate_kappa_for_target_spread` instead inverts
  the optimal-spread formula to find the `κ` that reproduces a *target*
  spread (e.g. a typical real NBBO spread) — a legitimate proxy technique,
  explicitly not a measurement.

**What calibrating honestly actually revealed, run on real AAPL data (252
days, `fill_size=100`):** the real numbers aren't close to the illustrative
defaults — calibrated `arrival_rate` came out ~2,112/bar against the
illustrative `1.0` (~2000x), and calibrated `κ` ~43 against the illustrative
`1.5` (~28x). Plugging the calibrated values into `historical.py`
(`dt=1` per bar) didn't produce a better backtest — it produced a **100% win
rate, bit-for-bit identical across every seed**, which is not success. At
`arrival_rate≈2112` and `dt=1`, `1 - exp(-A·dt)` saturates to ~1.0: a
*certainty* of a fill each day, not a probability, and since the loop only
allows one fill per side per bar, every run collapses to exactly 2 fills/day,
every day, with no randomness left for a seed to express. The calibrated
number is correct — AAPL's real volume means a touch-level quote would
certainly get hit within a day — the problem is that "one fill-or-no-fill
roll per side per bar" is too coarse a time resolution to represent that
liquidity at all. This is an honest, useful negative result: **the
daily-bar architecture (`historical.py`, the live engine) is structurally
suited to illustrating the model, not to realistically simulating a liquid
large-cap's microstructure** — that needs the synthetic simulator's
per-second `dt`, or real intraday data this repo doesn't have.

`time_horizon` in the live engine is a constant, refreshed every tick,
rather than a single terminal time — a live feed doesn't know in advance how
many bars remain. In effect: "size for unwinding this inventory over roughly
this many bars," not a single end-of-day liquidation. This is a deliberate
simplification of the paper's finite-horizon setup, not an oversight.

## Parameter reference

Every knob the dashboard's Market Maker tab (and the two CLIs,
`scripts/run_market_maker.py` and `python -m quant_pod.trader.cli quote`)
expose, grouped by what it actually controls.

**Core model shape** (the AS formulas themselves):

| Knob | What it does |
|---|---|
| `gamma` (risk aversion) | How hard you skew quotes away from mid while holding inventory. Higher = pushes back toward flat more aggressively. **Non-monotonic on spread width** — can widen *or* narrow it depending on `sigma`/`kappa` (see above) — but skew magnitude always grows with it. |
| `kappa` (order-arrival decay) | How fast your fill odds drop off as a quote moves away from mid. High = you basically have to be at the touch to trade; low = you can sit wider and still get hit. |
| `sigma` (volatility, price units) | Mid-price volatility in dollars. Drives spread quadratically and skew linearly. In historical/live mode this is *estimated* from real data (`vol_window` below) rather than set directly — watch for the price-level pitfall under Calibration if comparing across symbols/periods. |

**Flow — how often you trade:**

| Knob | What it does |
|---|---|
| `arrival_rate` (A) | Base rate of counterparties willing to trade at all. Scales fill *frequency* but, notably, doesn't appear in the optimal-spread formula at all — only `kappa` does. |
| `fill_size` | Shares per fill. PnL scales roughly linearly with this — the default (1 share) is why a multi-year real-symbol PnL total can look tiny; it's a position-sizing artifact, not a performance read (see "No transaction costs..." under Limitations). |
| `dt` / `horizon` (synthetic sim only) | Tick size and total simulated time; `horizon` decays to zero over the run (a true finite-horizon liquidation). |
| `time_horizon` (historical/live) | A *constant* sizing horizon, refreshed every tick, instead of decaying — a real/historical replay has no known end-of-day liquidation point. |

**Risk controls:**

| Knob | What it does |
|---|---|
| `max_inventory` | Hard position cap. Once a fill would breach it, that side's quote is pulled entirely (not widened) until inventory drifts back. |
| `quote_band` | Tolerance zone / hysteresis: only reposts a side once it's drifted past this threshold from what's currently quoted, instead of snapping every tick. Trades pricing precision for far fewer quote updates — see `avellaneda_stoikov.apply_tolerance_band`. |

**Mechanics / bookkeeping:**

| Knob | What it does |
|---|---|
| `mid0` (synthetic only) | Starting price. |
| `vol_window`, `default_sigma` (historical/live) | Rolling window (bars) used to estimate `sigma` from real data, and the fallback until that window fills. |
| `initial_cash` | Starting capital. |
| `seed` | Random seed for the fill-decision draws (and the synthetic price path, if applicable) — same seed, same run. |
| `risk_limits` (live only) | Reuses the pod's real `RiskLimits` class for the pre-trade inventory check, instead of the simpler `max_inventory` cutoff used elsewhere. |

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
- **No transaction costs, and "Total PnL" is a raw dollar sum, not a
  return.** Every fill is `fill_size` shares (1 by default), so PnL scales
  roughly linearly with that one knob — a few hundred dollars from a
  multi-year real-symbol run isn't a weak result, it's 1-share-at-a-time
  trading with no capital-base or return-on-capital concept anywhere in the
  code.
- **The "it basically never loses" finding was partly an artifact of the
  mid-crossing bug above, not a free lunch.** Before
  `suppress_quotes_crossing_mid` existed: a stress test on the *synthetic*
  simulator (small illustrative `sigma=0.3`, which never gets near the
  mid-crossing threshold) showed a 100% win rate across 300 seeds, even with
  weak inventory control and a 20x longer horizon — because every fill
  captures a guaranteed positive spread while unresolved inventory risk only
  grows like the square root of ticks, so the guaranteed edge dominates for
  long enough runs. That argument is still correct, *conditional on quotes
  staying sane*. It does **not** hold once real dollar-scale `sigma` pushes
  skew past the spread: re-running the same kind of sweep on real AAPL data
  (where sigma is large enough to approach that threshold) gives a **46-54%
  win rate with genuine losses**, both before and after the fix — before the
  fix those losses were catastrophic and guaranteed (every single run with
  `fill_size > 1` lost money, because every large position triggered a
  self-destructive crossed quote); after the fix they're real but bounded to
  actual price risk. Moral: the "guaranteed edge beats noise" argument is a
  property of a well-behaved quoting model, not a law of nature — it breaks
  if the model itself is broken. Transaction costs remain a real, separate
  gap: there still isn't a fee that could make an individual fill
  value-negative, only inventory risk can.
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
