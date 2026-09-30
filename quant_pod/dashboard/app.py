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
from quant_pod.common.config import DATA_DIR
from quant_pod.ingester import store
from quant_pod.ingester.sources.yfinance_source import YFinanceSource
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


st.title("quant-pod research dashboard")

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
