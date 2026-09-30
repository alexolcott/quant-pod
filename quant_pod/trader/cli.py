"""`python -m quant_pod.trader.cli AAPL --strategy moving_average --engine python`"""
from __future__ import annotations

import asyncio
import sys

import typer

from quant_pod.analyzer.report import generate_report
from quant_pod.common.logging import get_logger
from quant_pod.trader.engine import run_trader

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


if __name__ == "__main__":
    app()
