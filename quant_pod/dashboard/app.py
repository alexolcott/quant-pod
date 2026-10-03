"""Interactive research dashboard: pick a symbol + strategy, run a backtest,
and inspect the equity curve, metrics, and trade log. Reuses the same
ingester/research/analyzer code the CLI and live trader use -- this is a
viewer on top of that pipeline, not a separate implementation.

Run with: streamlit run quant_pod/dashboard/app.py
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quant_pod.analyzer import metrics as m
from quant_pod.common.config import DATA_DIR, REPO_ROOT
from quant_pod.ingester import store
from quant_pod.ingester.sources.yfinance_source import YFinanceSource
from quant_pod.marketmaking.avellaneda_stoikov import AvellanedaStoikovQuoter
from quant_pod.marketmaking.historical import run_market_maker_on_bars
from quant_pod.marketmaking.markout import compute_markouts, summarize_markouts
from quant_pod.marketmaking.simulator import run_market_maker_sim
from quant_pod.marketmaking.simulator import summarize as summarize_market_maker
from quant_pod.research.backtester import run_backtest
from quant_pod.research.strategies.moving_average import MovingAverageCrossover
from quant_pod.research.strategies.risk_ratio import RiskRatioStrategy

# Reference palette (validated colorblind-safe categorical order + chart chrome),
# see quant-pod/README or the dataviz skill's references/palette.md.
COLOR_SERIES_1 = "#2a78d6"  # blue -- single-strategy line, "strategy A" in comparisons
COLOR_SERIES_2 = "#eb6834"  # orange -- "strategy B" in comparisons
COLOR_DRAWDOWN = "#e34948"  # red -- underwater/loss magnitude
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"
COLOR_MUTED = "#898781"

STRATEGIES = {
    "moving_average": MovingAverageCrossover,
    "risk_ratio": RiskRatioStrategy,
}

st.set_page_config(page_title="quant-pod research dashboard", layout="wide")


def cached_symbols() -> list[str]:
    return sorted(p.stem for p in DATA_DIR.glob("*.parquet"))


def _load_doc_section(path, heading: str) -> str:
    """Extracts the text between a '## {heading}' line and the next '## '
    heading (or end of file) from a markdown doc -- so the in-app popup and
    docs/MARKET_MAKING.md stay a single source of truth instead of two copies
    that can drift apart.
    """
    text = path.read_text(encoding="utf-8")
    marker = f"## {heading}"
    start = text.index(marker) + len(marker)
    rest = text[start:]
    end = rest.find("\n## ")
    return rest[: end if end != -1 else None].strip()


@st.dialog("Market maker parameter reference", width="large")
def _show_parameter_reference():
    doc_path = REPO_ROOT / "docs" / "MARKET_MAKING.md"
    try:
        st.markdown(_load_doc_section(doc_path, "Parameter reference"))
    except (FileNotFoundError, ValueError):
        st.error("Couldn't load the parameter reference from docs/MARKET_MAKING.md.")


def equity_figure(equity_curves: dict[str, pd.Series]) -> go.Figure:
    colors = [COLOR_SERIES_1, COLOR_SERIES_2]
    fig = go.Figure()
    for (name, curve), color in zip(equity_curves.items(), colors):
        fig.add_trace(
            go.Scatter(
                x=curve.index, y=curve.values, mode="lines", name=name,
                line=dict(color=color, width=2),
                hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>" + name + "</extra>",
            )
        )
    fig.update_layout(
        title="Equity Curve",
        showlegend=len(equity_curves) > 1,
        margin=dict(l=40, r=20, t=40, b=30),
        height=380,
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        xaxis=dict(gridcolor=COLOR_GRID, linecolor=COLOR_AXIS),
        yaxis=dict(title="Equity ($)", gridcolor=COLOR_GRID, linecolor=COLOR_AXIS),
    )
    return fig


def drawdown_figure(equity: pd.Series) -> go.Figure:
    running_max = equity.cummax()
    drawdown = (equity / running_max - 1) * 100
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdown.index, y=drawdown.values, mode="lines", fill="tozeroy",
            line=dict(color=COLOR_DRAWDOWN, width=1.5),
            fillcolor="rgba(227, 73, 72, 0.15)",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra>drawdown</extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        title="Drawdown",
        margin=dict(l=40, r=20, t=40, b=30),
        height=200,
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        xaxis=dict(gridcolor=COLOR_GRID, linecolor=COLOR_AXIS),
        yaxis=dict(title="%", gridcolor=COLOR_GRID, linecolor=COLOR_AXIS),
    )
    return fig


def mm_price_figure(result, x=None, x_title: str = "step") -> go.Figure:
    x = x if x is not None else list(range(len(result.mid)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=result.ask, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(
        go.Scatter(
            x=x, y=result.bid, mode="lines", line=dict(width=0), fill="tonexty",
            fillcolor="rgba(42, 120, 214, 0.15)", hoverinfo="skip", showlegend=False, name="bid-ask",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=result.mid, mode="lines", name="mid",
            line=dict(color=COLOR_SERIES_1, width=1.5),
            hovertemplate=f"{x_title} %{{x}}<br>mid $%{{y:.2f}}<extra>mid</extra>",
        )
    )
    fig.update_layout(
        title="Quotes around mid price (shaded band = bid/ask; gaps = inventory limit pulled that side)",
        showlegend=False,
        margin=dict(l=40, r=20, t=40, b=30),
        height=340,
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        xaxis=dict(title=x_title, gridcolor=COLOR_GRID, linecolor=COLOR_AXIS),
        yaxis=dict(title="Price ($)", gridcolor=COLOR_GRID, linecolor=COLOR_AXIS),
    )
    return fig


def mm_inventory_figure(result, x=None, x_title: str = "step") -> go.Figure:
    x = x if x is not None else list(range(len(result.inventory)))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=result.inventory, mode="lines", fill="tozeroy",
            line=dict(color=COLOR_SERIES_2, width=1.5), fillcolor="rgba(235, 104, 52, 0.15)",
            hovertemplate=f"{x_title} %{{x}}<br>inventory %{{y:.0f}}<extra>inventory</extra>", showlegend=False,
        )
    )
    fig.update_layout(
        title="Inventory",
        margin=dict(l=40, r=20, t=40, b=30),
        height=200,
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        xaxis=dict(title=x_title, gridcolor=COLOR_GRID, linecolor=COLOR_AXIS),
        yaxis=dict(title="shares", gridcolor=COLOR_GRID, linecolor=COLOR_AXIS, zeroline=True, zerolinecolor=COLOR_AXIS),
    )
    return fig


MARKOUT_HORIZONS = [10, 60, 300]  # ~10s / 1min / 5min at the simulator's default dt (one simulated second)


st.title("quant-pod research dashboard")

mode = st.sidebar.radio("Mode", ["Bar strategies", "Market maker"])
st.sidebar.divider()

if mode == "Market maker":
    with st.sidebar:
        if st.button("📖 Parameter reference"):
            _show_parameter_reference()

        st.header("Data")
        data_source = st.radio("Data source", ["Synthetic", "Real symbol"])

        if data_source == "Real symbol":
            mm_symbols = cached_symbols()
            mm_symbol = st.selectbox(
                "Symbol", options=mm_symbols, index=0 if mm_symbols else None,
                placeholder="No data cached yet", key="mm_symbol",
            )
            mm_start = st.date_input("Start", value=date.today() - timedelta(days=365), key="mm_start")
            mm_end = st.date_input("End", value=date.today(), key="mm_end")
        else:
            mid0 = st.number_input("Starting mid-price", value=100.0)
            sigma = st.slider("Mid-price volatility (sigma)", 0.05, 2.0, 0.3, step=0.05)
            horizon = st.slider("Horizon (trading days)", 0.1, 3.0, 1.0, step=0.1)

        st.header("Model")
        gamma = st.slider("Risk aversion (gamma)", 0.0, 2.0, 0.1, step=0.01)
        kappa = st.slider("Order-arrival decay (kappa)", 0.5, 5.0, 1.5, step=0.1)
        if data_source == "Real symbol":
            # Per-BAR intensity/horizon here, not per-second like the synthetic
            # side's defaults -- real bars are daily, so those values would
            # saturate (arrival_rate=140/day would mean ~140 fills/day) or
            # decay to nothing (a 1-day horizon vanishes after bar 1) if reused.
            arrival_rate = st.slider("Arrival intensity (A, per bar)", 0.1, 20.0, 1.0, step=0.1)
            time_horizon = st.slider("Time horizon (bars, constant)", 1.0, 60.0, 20.0, step=1.0)
            vol_window = st.slider("Volatility window (bars)", 5, 60, 20)
            default_sigma = st.number_input("Fallback sigma (used until the window fills)", value=1.0)
        else:
            arrival_rate = st.slider("Arrival intensity (A)", 1.0, 300.0, 140.0)

        fill_size = st.number_input(
            "Shares per fill", min_value=0.01, value=1.0, step=1.0,
            help="PnL scales roughly linearly with this -- the default (1 share) is why multi-year PnL can look tiny.",
        )

        limit_inventory = st.checkbox("Limit inventory", value=False)
        max_inventory = (
            st.slider("Max inventory (shares, like 'shares per fill' above)", 1.0, 50.0, 10.0, step=1.0)
            if limit_inventory else None
        )

        use_quote_band = st.checkbox("Tolerance zone (reduce requoting)", value=False)
        quote_band = (
            st.slider(
                "Band (price units)", 0.01, 2.0, 0.1, step=0.01,
                help="Only repost a side once it drifts more than this from what's currently posted.",
            )
            if use_quote_band else None
        )

        seed = st.number_input("Random seed", value=7, step=1)
        run_mm = st.button("Run simulation", type="primary")

    if not run_mm and "last_mm_result" not in st.session_state:
        st.info("Configure the model in the sidebar and click **Run simulation**.")
        st.stop()

    if run_mm:
        if data_source == "Real symbol":
            if not mm_symbol:
                st.error("Fetch a symbol in Bar strategies mode first, or pick a cached one.")
                st.stop()
            bars = store.read_bars(mm_symbol, start=mm_start, end=mm_end)
            if bars.empty:
                st.error(f"No cached {mm_symbol} bars between {mm_start} and {mm_end}.")
                st.stop()
            with st.spinner("Simulating..."):
                result = run_market_maker_on_bars(
                    bars, gamma=gamma, kappa=kappa, arrival_rate=arrival_rate, fill_size=fill_size,
                    time_horizon=time_horizon, vol_window=vol_window, default_sigma=default_sigma,
                    max_inventory=max_inventory, quote_band=quote_band, seed=int(seed),
                )
            result_x, result_x_title = bars.index, "date"
        else:
            quoter = AvellanedaStoikovQuoter(gamma=gamma, kappa=kappa, sigma=sigma)
            with st.spinner("Simulating..."):
                result = run_market_maker_sim(
                    quoter, mid0=mid0, sigma=sigma, horizon=horizon, arrival_rate=arrival_rate, fill_size=fill_size,
                    max_inventory=max_inventory, quote_band=quote_band, seed=int(seed),
                )
            result_x, result_x_title = None, "step"
        st.session_state["last_mm_result"] = (result, result_x, result_x_title)

    result, result_x, result_x_title = st.session_state["last_mm_result"]
    stats = summarize_market_maker(result)

    st.subheader("Avellaneda-Stoikov market maker")
    cols = st.columns(6)
    cols[0].metric("Total PnL", f"{stats['total_pnl']:+.2f}")
    cols[1].metric("Fills", f"{stats['num_fills']}")
    cols[2].metric("Final inventory", f"{stats['final_inventory']:+.0f}")
    cols[3].metric("Max |inventory|", f"{stats['max_abs_inventory']:.0f}")
    cols[4].metric("Avg spread", f"{stats['avg_spread_bps']:.1f} bps")
    cols[5].metric("Requote rate", f"{stats['bid_requote_rate']:.0%}")

    st.plotly_chart(mm_price_figure(result, x=result_x, x_title=result_x_title), use_container_width=True)
    st.plotly_chart(mm_inventory_figure(result, x=result_x, x_title=result_x_title), use_container_width=True)

    st.subheader("Markout decomposition")
    if not result.fills:
        st.caption("No fills.")
    else:
        horizons = [h for h in MARKOUT_HORIZONS if h < len(result.mid)]
        markouts = compute_markouts(result, horizons=horizons)
        markout_stats = summarize_markouts(markouts, horizons=horizons)
        table = pd.DataFrame(markout_stats).T.rename_axis("horizon").rename(
            columns={
                "avg_spread_capture": "spread capture", "avg_adverse_selection": "adverse selection",
                "avg_total_markout": "total", "n": "n fills",
            }
        )
        st.caption("Per-unit, averaged over fills with a full horizon available.")
        st.dataframe(
            table.style.format({
                "spread capture": "{:+.4f}", "adverse selection": "{:+.4f}",
                "total": "{:+.4f}", "n fills": "{:.0f}",
            }),
            use_container_width=True,
        )
    st.stop()

with st.sidebar:
    st.header("Data")
    symbols = cached_symbols()
    symbol = st.selectbox("Symbol", options=symbols, index=0 if symbols else None, placeholder="No data cached yet")

    with st.expander("Fetch a new symbol"):
        new_symbol = st.text_input("Ticker", value="AAPL").upper().strip()
        start = st.date_input("Start", value=date.today() - timedelta(days=365 * 5))
        end = st.date_input("End", value=date.today())
        if st.button("Fetch"):
            with st.spinner(f"Fetching {new_symbol}..."):
                df = YFinanceSource().fetch(new_symbol, start, end)
                if df.empty:
                    st.error(f"No data returned for {new_symbol!r}.")
                else:
                    store.write_bars(new_symbol, df)
                    st.success(f"Stored {len(df)} rows for {new_symbol}.")
                    st.rerun()

    st.header("Strategy")
    strategy_name = st.selectbox("Strategy", options=list(STRATEGIES))

    if strategy_name == "moving_average":
        fast = st.slider("Fast window", 5, 50, 20)
        slow = st.slider("Slow window", 20, 200, 50)
        position_size = st.number_input("Position size (shares)", value=100.0)
        strategy_params = dict(fast=fast, slow=slow, position_size=position_size)
    else:
        window = st.slider("Rolling window", 20, 120, 60)
        buy_threshold = st.slider("Buy threshold (z-score)", -3.0, 0.0, -1.0, step=0.1)
        position_size = st.number_input("Position size (shares)", value=100.0)
        strategy_params = dict(window=window, buy_threshold=buy_threshold, position_size=position_size)

    compare_all = st.checkbox("Compare all strategies (default params)", value=False)
    run = st.button("Run backtest", type="primary", disabled=symbol is None)

if symbol is None:
    st.info("Fetch a symbol in the sidebar to get started.")
    st.stop()

if not run and "last_result" not in st.session_state:
    st.info("Configure a strategy in the sidebar and click **Run backtest**.")
    st.stop()

bars = store.read_bars(symbol)

if run:
    if compare_all:
        results = {name: run_backtest(cls(), bars) for name, cls in STRATEGIES.items()}
    else:
        strategy = STRATEGIES[strategy_name](**strategy_params)
        results = {strategy.strategy_id: run_backtest(strategy, bars)}
    st.session_state["last_result"] = (symbol, results)

symbol, results = st.session_state["last_result"]

if len(results) > 1:
    rows = {name: m.summarize(r.equity_curve, r.returns, r.trades) for name, r in results.items()}
    st.subheader(f"{symbol} — strategy comparison")
    st.dataframe(pd.DataFrame(rows).T.style.format({
        "total_return": "{:.2%}", "cagr": "{:.2%}", "sharpe": "{:.2f}", "sortino": "{:.2f}",
        "max_drawdown": "{:.2%}", "calmar": "{:.2f}", "var_95": "{:.2%}", "turnover": "{:.2f}x",
        "num_trades": "{:.0f}",
    }))
    st.plotly_chart(equity_figure({name: r.equity_curve for name, r in results.items()}), use_container_width=True)
else:
    name, result = next(iter(results.items()))
    summary = m.summarize(result.equity_curve, result.returns, result.trades)

    st.subheader(f"{symbol} — {name}")
    cols = st.columns(6)
    cols[0].metric("Total return", f"{summary['total_return']:.2%}")
    cols[1].metric("CAGR", f"{summary['cagr']:.2%}")
    cols[2].metric("Sharpe", f"{summary['sharpe']:.2f}")
    cols[3].metric("Sortino", f"{summary['sortino']:.2f}")
    cols[4].metric("Max drawdown", f"{summary['max_drawdown']:.2%}")
    cols[5].metric("Trades", f"{summary['num_trades']:.0f}")

    st.plotly_chart(equity_figure({name: result.equity_curve}), use_container_width=True)
    st.plotly_chart(drawdown_figure(result.equity_curve), use_container_width=True)

    st.subheader("Trade log")
    if result.trades.empty:
        st.caption("No trades.")
    else:
        st.dataframe(result.trades, use_container_width=True)
