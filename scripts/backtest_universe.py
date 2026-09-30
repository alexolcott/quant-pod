"""Run every strategy against every cached symbol and rank the results.

Each symbol is still backtested independently (this is a per-symbol screen,
not a portfolio strategy that allocates capital across names at once -- see
docs/GUIDE.md for that distinction). What this answers: "across the whole
S&P 500, which names does this strategy actually work on, and how does one
strategy compare to another on average?"

Usage:
    python scripts/backtest_universe.py
    python scripts/backtest_universe.py --strategy moving_average --top 20
"""
from __future__ import annotations

import sys
import time

import pandas as pd
import typer

from quant_pod.analyzer import metrics as m
from quant_pod.common.config import DATA_DIR, REPORTS_DIR
from quant_pod.common.logging import get_logger
from quant_pod.ingester import store
from quant_pod.research.backtester import run_backtest
from quant_pod.research.strategies.moving_average import MovingAverageCrossover
from quant_pod.research.strategies.risk_ratio import RiskRatioStrategy

log = get_logger("backtest_universe")
app = typer.Typer(add_completion=False)

STRATEGIES = {
    "moving_average": MovingAverageCrossover,
    "risk_ratio": RiskRatioStrategy,
}


@app.command()
def main(
    strategy: str = typer.Option("all", help="moving_average | risk_ratio | all"),
    top: int = typer.Option(15, help="How many best/worst performers to print per strategy"),
    min_rows: int = typer.Option(100, help="Skip symbols with fewer stored bars than this"),
):
    symbols = sorted(p.stem for p in DATA_DIR.glob("*.parquet"))
    if not symbols:
        log.error("No cached symbols found. Run the ingester (or scripts/fetch_sp500.py) first.")
        raise typer.Exit(code=1)

    strategy_names = list(STRATEGIES) if strategy == "all" else [strategy]
    for name in strategy_names:
        if name not in STRATEGIES:
            raise typer.BadParameter(f"strategy must be one of {list(STRATEGIES)} or 'all'")

    log.info("Backtesting %d symbol(s) x %d strateg(ies)...", len(symbols), len(strategy_names))
    t0 = time.perf_counter()

    all_rows: list[dict] = []
    skipped = 0
    for strategy_name in strategy_names:
        strategy_cls = STRATEGIES[strategy_name]
        for symbol in symbols:
            bars = store.read_bars(symbol)
            if len(bars) < min_rows:
                skipped += 1
                continue
            try:
                result = strategy_cls()
                bt = run_backtest(result, bars)
                summary = m.summarize(bt.equity_curve, bt.returns, bt.trades)
                summary["symbol"] = symbol
                summary["strategy"] = strategy_name
                all_rows.append(summary)
            except Exception as e:
                log.warning("%s/%s failed: %s", strategy_name, symbol, e)

    elapsed = time.perf_counter() - t0
    if not all_rows:
        log.error("No backtests completed.")
        raise typer.Exit(code=1)

    df = pd.DataFrame(all_rows).set_index(["strategy", "symbol"])
    out_path = REPORTS_DIR / "universe_backtest.csv"
    df.to_csv(out_path)
    log.info(
        "Ran %d backtests (%d symbols skipped for insufficient history) in %.1fs. Full results: %s",
        len(all_rows), skipped, elapsed, out_path,
    )

    for strategy_name in strategy_names:
        sub = df.loc[strategy_name]
        print(f"\n{'=' * 60}\n{strategy_name} -- {len(sub)} symbols\n{'=' * 60}")
        print(f"Universe average:  total_return={sub['total_return'].mean():.2%}  "
              f"sharpe={sub['sharpe'].mean():.2f}  max_drawdown={sub['max_drawdown'].mean():.2%}")
        print(f"Universe median:   total_return={sub['total_return'].median():.2%}  "
              f"sharpe={sub['sharpe'].median():.2f}")
        win_rate = (sub["total_return"] > 0).mean()
        print(f"Win rate (positive return): {win_rate:.1%}")

        print(f"\nTop {top} by Sharpe:")
        print(sub.sort_values("sharpe", ascending=False)[["total_return", "sharpe", "max_drawdown", "num_trades"]]
              .head(top).to_string(formatters={"total_return": "{:.2%}".format, "max_drawdown": "{:.2%}".format}))

        print(f"\nBottom {top} by Sharpe:")
        print(sub.sort_values("sharpe", ascending=True)[["total_return", "sharpe", "max_drawdown", "num_trades"]]
              .head(top).to_string(formatters={"total_return": "{:.2%}".format, "max_drawdown": "{:.2%}".format}))


if __name__ == "__main__":
    app()
