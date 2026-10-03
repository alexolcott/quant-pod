"""Run the Avellaneda-Stoikov market-making simulation and print a summary.

Usage:
    python scripts/run_market_maker.py
    python scripts/run_market_maker.py --gamma 0.5 --kappa 1.5 --seed 42
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import typer

from quant_pod.analyzer.robust_stats import block_bootstrap_sharpe_ci
from quant_pod.common.logging import get_logger
from quant_pod.marketmaking.avellaneda_stoikov import AvellanedaStoikovQuoter
from quant_pod.marketmaking.markout import compute_markouts, format_markout_table, summarize_markouts
from quant_pod.marketmaking.simulator import run_market_maker_sim, summarize

log = get_logger("run_market_maker")
app = typer.Typer(add_completion=False)

MARKOUT_HORIZONS = [10, 60, 300]  # ~10s, 1min, 5min at dt=1 simulated second


@app.command()
def main(
    gamma: float = typer.Option(0.1, help="Risk aversion"),
    kappa: float = typer.Option(1.5, help="Order-arrival intensity decay"),
    sigma: float = typer.Option(0.3, help="Mid-price volatility (price units)"),
    mid0: float = typer.Option(100.0, help="Starting mid-price"),
    arrival_rate: float = typer.Option(140.0, help="'A' in lambda(delta) = A * exp(-kappa * delta)"),
    horizon: float = typer.Option(1.0, help="Simulated horizon, in trading days"),
    fill_size: float = typer.Option(1.0, help="Shares per fill -- PnL scales roughly linearly with this"),
    max_inventory: float | None = typer.Option(None, help="Hard position limit, in shares like fill_size (unbounded if unset)"),
    quote_band: float | None = typer.Option(None, help="Tolerance zone: only repost a side once it drifts past this (unset = requote every tick)"),
    seed: int = typer.Option(7, help="Random seed"),
):
    quoter = AvellanedaStoikovQuoter(gamma=gamma, kappa=kappa, sigma=sigma)
    result = run_market_maker_sim(
        quoter, mid0=mid0, sigma=sigma, horizon=horizon, arrival_rate=arrival_rate, fill_size=fill_size,
        max_inventory=max_inventory, quote_band=quote_band, seed=seed,
    )
    stats = summarize(result)

    log.info("Simulated %d steps (gamma=%.3f kappa=%.2f sigma=%.2f A=%.1f)", len(result.mid) - 1, gamma, kappa, sigma, arrival_rate)
    print(f"Total PnL:            {stats['total_pnl']:+.2f}")
    print(f"Fills:                {stats['num_fills']}")
    print(f"Final inventory:      {stats['final_inventory']:+.1f}")
    print(f"Max |inventory|:      {stats['max_abs_inventory']:.1f}")
    print(f"Inventory std dev:    {stats['inventory_std']:.2f}")
    print(f"Avg quoted spread:    {stats['avg_spread_bps']:.1f} bps")
    print(f"Requote rate (bid/ask): {stats['bid_requote_rate']:.1%} / {stats['ask_requote_rate']:.1%}")
    if max_inventory is not None:
        print(f"Time quoting bid/ask: {stats['pct_steps_quoting_bid']:.1%} / {stats['pct_steps_quoting_ask']:.1%}")

    if result.fills:
        markouts = compute_markouts(result, horizons=MARKOUT_HORIZONS)
        markout_stats = summarize_markouts(markouts, horizons=MARKOUT_HORIZONS)
        print("\nMarkout decomposition (per-unit, avg over fills with a full horizon):")
        print(format_markout_table(markout_stats, MARKOUT_HORIZONS))

    # PnL increments, not percentage returns: equity starts at/near 0 with the
    # default initial_cash, so pct_change() is ill-defined here -- and a
    # dollar-PnL book's Sharpe is conventionally computed on PnL increments
    # directly anyway. periods_per_year=1 because this is a single simulated
    # day's path, not a repeated-period series -- the number below is a
    # per-simulation Sharpe, deliberately not annualized.
    pnl_increments = pd.Series(np.diff(result.equity))
    ci = block_bootstrap_sharpe_ci(pnl_increments, n_bootstrap=2000, block_size=300, periods_per_year=1, seed=seed)
    print(
        f"\nPnL Sharpe (per-simulation, not annualized), 95% block-bootstrap CI: "
        f"{ci['point_estimate']:.3f}  [{ci['lower']:.3f}, {ci['upper']:.3f}]"
    )


if __name__ == "__main__":
    app()
