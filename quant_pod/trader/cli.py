"""`python -m quant_pod.trader.cli AAPL --strategy moving_average --engine python`"""
from __future__ import annotations

import asyncio
import sys

import typer

from quant_pod.analyzer.report import generate_report
from quant_pod.common.logging import get_logger
from quant_pod.marketmaking.markout import compute_markouts, format_markout_table, summarize_markouts
from quant_pod.marketmaking.simulator import summarize as summarize_market_maker
from quant_pod.trader.engine import run_trader
from quant_pod.trader.market_maker_engine import run_market_maker_trader
from quant_pod.trader.risk_limits import RiskLimits

log = get_logger("trader.cli")

if sys.platform == "win32":
    # zmq.asyncio needs add_reader/add_writer, which Windows' default
    # ProactorEventLoop doesn't implement.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

STRATEGIES = {}


def _load_strategies():
    if STRATEGIES:
        return STRATEGIES
    from quant_pod.research.strategies.moving_average import MovingAverageCrossover
    from quant_pod.research.strategies.risk_ratio import RiskRatioStrategy

    STRATEGIES["moving_average"] = MovingAverageCrossover
    STRATEGIES["risk_ratio"] = RiskRatioStrategy
    return STRATEGIES


app = typer.Typer(add_completion=False)


@app.command()
def trade(
    symbol: str,
    strategy: str = typer.Option("moving_average", help="moving_average | risk_ratio"),
    engine: str = typer.Option("python", help="python | cpp"),
    report: bool = typer.Option(True, help="Generate an analyzer report after the run"),
):
    strategies = _load_strategies()
    if strategy not in strategies:
        raise typer.BadParameter(f"strategy must be one of {list(strategies)}")

    strat = strategies[strategy]()
    result = asyncio.run(run_trader(symbol, strat, engine=engine))

    if report:
        generate_report(f"{symbol} {strategy} (live/{engine})", result.equity_curve, result.returns, result.trades)


@app.command()
def quote(
    symbol: str,
    gamma: float = typer.Option(0.1, help="Risk aversion"),
    kappa: float = typer.Option(1.5, help="Order-arrival intensity decay"),
    arrival_rate: float = typer.Option(1.0, help="'A' in lambda(delta) = A * exp(-kappa * delta), per bar"),
    time_horizon: float = typer.Option(1.0, help="Constant AS time-to-horizon term, refreshed every tick"),
    max_position: float = typer.Option(10.0, help="Hard inventory limit (reuses RiskLimits.max_position)"),
    vol_window: int = typer.Option(20, help="Rolling window (in bars) for the live sigma estimate"),
    default_sigma: float = typer.Option(1.0, help="Sigma to use until vol_window bars of history accumulate"),
    initial_cash: float = typer.Option(100_000.0, help="Starting cash"),
    seed: int = typer.Option(7, help="Random seed for the fill-decision draws"),
):
    """Run the Avellaneda-Stoikov market maker live against the replay feed
    (start `python -m quant_pod.ingester.replay SYMBOL` first)."""
    result = asyncio.run(
        run_market_maker_trader(
            symbol, gamma=gamma, kappa=kappa, arrival_rate=arrival_rate, time_horizon=time_horizon,
            vol_window=vol_window, default_sigma=default_sigma, initial_cash=initial_cash,
            risk_limits=RiskLimits(max_position=max_position), seed=seed,
        )
    )

    stats = summarize_market_maker(result)
    print(f"Total PnL:       {stats['total_pnl']:+.2f}")
    print(f"Fills:           {stats['num_fills']}")
    print(f"Final inventory: {stats['final_inventory']:+.1f}")
    print(f"Max |inventory|: {stats['max_abs_inventory']:.1f}")

    if result.fills:
        horizons = [h for h in (1, 5, 20) if h < len(result.mid)]
        markouts = compute_markouts(result, horizons=horizons)
        markout_stats = summarize_markouts(markouts, horizons=horizons)
        print("\nMarkout decomposition (per-unit, in bars):")
        print(format_markout_table(markout_stats, horizons))


if __name__ == "__main__":
    app()
