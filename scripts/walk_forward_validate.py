"""Purged walk-forward validation: pick each strategy's hyperparameters using
only data before a test fold, score out-of-sample on that fold, and compare
against the Sharpe you'd report by optimizing hyperparameters on the whole
history at once (what a naive single backtest does). The gap between the two
is a concrete measure of how much of a single-backtest Sharpe is overfitting
to that one sample path rather than a strategy that generalizes.

Usage:
    python scripts/walk_forward_validate.py AAPL --strategy moving_average
    python scripts/walk_forward_validate.py AAPL --strategy risk_ratio --n-splits 8
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import typer

from quant_pod.analyzer import metrics as m
from quant_pod.analyzer.robust_stats import purged_walk_forward_splits
from quant_pod.common.logging import get_logger
from quant_pod.ingester import store
from quant_pod.research.backtester import run_backtest
from quant_pod.research.strategies.moving_average import MovingAverageCrossover
from quant_pod.research.strategies.risk_ratio import RiskRatioStrategy

log = get_logger("walk_forward_validate")
app = typer.Typer(add_completion=False)

STRATEGIES = {
    "moving_average": MovingAverageCrossover,
    "risk_ratio": RiskRatioStrategy,
}

PARAM_GRIDS = {
    "moving_average": [{"fast": f, "slow": s} for f in (10, 20, 30) for s in (50, 100, 150) if f < s],
    "risk_ratio": [{"window": w, "buy_threshold": t} for w in (30, 60, 90) for t in (-1.5, -1.0, -0.5)],
}


def _best_params(strategy_cls, param_grid: list[dict], bars: pd.DataFrame) -> tuple[dict, float]:
    """Backtests every grid point on `bars` and returns the best by Sharpe."""
    best_params, best_sharpe = param_grid[0], -np.inf
    for params in param_grid:
        bt = run_backtest(strategy_cls(**params), bars)
        sharpe = m.sharpe_ratio(bt.returns)
        if sharpe > best_sharpe:
            best_params, best_sharpe = params, sharpe
    return best_params, best_sharpe


@app.command()
def main(
    symbol: str = typer.Argument("AAPL"),
    strategy: str = typer.Option("moving_average", help="moving_average | risk_ratio"),
    n_splits: int = typer.Option(5, help="Number of walk-forward folds"),
    embargo: int = typer.Option(5, help="Bars purged from training immediately before each test fold"),
):
    if strategy not in STRATEGIES:
        raise typer.BadParameter(f"strategy must be one of {list(STRATEGIES)}")
    strategy_cls = STRATEGIES[strategy]
    param_grid = PARAM_GRIDS[strategy]

    bars = store.read_bars(symbol)
    if bars.empty:
        log.error("No cached data for %s. Fetch it first.", symbol)
        raise typer.Exit(code=1)

    splits = purged_walk_forward_splits(len(bars), n_splits, embargo)

    log.info("%s/%s: %d folds over %d bars (embargo=%d)", symbol, strategy, n_splits, len(bars), embargo)
    print(f"\n{'fold':>4s} {'train params':>28s} {'train sharpe':>13s} {'test sharpe (OOS)':>18s}")

    all_oos_returns = []
    for i, (train_idx, test_idx) in enumerate(splits, start=1):
        if len(train_idx) == 0:
            log.warning("Fold %d has no training data (embargo too large) -- skipping.", i)
            continue

        train_bars = bars.iloc[: train_idx[-1] + 1]
        params, train_sharpe = _best_params(strategy_cls, param_grid, train_bars)

        test_window = bars.iloc[: test_idx[-1] + 1]
        test_bt = run_backtest(strategy_cls(**params), test_window)
        oos_returns = test_bt.returns.iloc[test_idx]
        oos_sharpe = m.sharpe_ratio(oos_returns)
        all_oos_returns.append(oos_returns)

        print(f"{i:>4d} {str(params):>28s} {train_sharpe:>13.2f} {oos_sharpe:>18.2f}")

    if not all_oos_returns:
        log.error("No folds produced results -- try fewer splits or a smaller embargo.")
        raise typer.Exit(code=1)

    pooled_oos = pd.concat(all_oos_returns)
    pooled_oos_sharpe = m.sharpe_ratio(pooled_oos)

    naive_params, naive_sharpe = _best_params(strategy_cls, param_grid, bars)

    print(f"\nPooled out-of-sample Sharpe across all folds:        {pooled_oos_sharpe:.2f}")
    print(f"Naive whole-history-optimized Sharpe (params={naive_params}): {naive_sharpe:.2f}")
    print(
        f"Overfitting gap (naive - pooled OOS):                {naive_sharpe - pooled_oos_sharpe:+.2f} "
        "-- how much of the naive single-backtest Sharpe doesn't survive out-of-sample"
    )


if __name__ == "__main__":
    app()
