"""Runs the whole pod end-to-end: ingest (if needed) -> replay publisher
(subprocess) -> trader subscriber (subprocess) -> analyzer report.

Usage:
    python scripts/run_pod.py --symbol AAPL --strategy moving_average --engine python
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime, timedelta

import typer

from quant_pod.common.logging import get_logger
from quant_pod.ingester import store

log = get_logger("run_pod")
app = typer.Typer(add_completion=False)

PYTHON = sys.executable


@app.command()
def main(
    symbol: str = typer.Argument("AAPL"),
    strategy: str = typer.Option("moving_average", help="moving_average | risk_ratio"),
    engine: str = typer.Option("python", help="python | cpp"),
    years: int = typer.Option(5, help="Years of history to ensure are ingested"),
    replay_speed: float = typer.Option(0.0, help="Seconds between replayed bars (0 = max speed)"),
):
    if not store.has_symbol(symbol):
        end = datetime.today()
        start = end - timedelta(days=365 * years)
        log.info("No cached data for %s, fetching %d years via yfinance...", symbol, years)
        subprocess.run(
            [
                PYTHON, "-m", "quant_pod.ingester.cli", "fetch", symbol,
                "--start", start.strftime("%Y-%m-%d"),
                "--end", end.strftime("%Y-%m-%d"),
                "--source", "yfinance",
            ],
            check=True,
        )

    log.info("Starting trader subprocess (symbol=%s strategy=%s engine=%s)", symbol, strategy, engine)
    trader_proc = subprocess.Popen(
        [PYTHON, "-m", "quant_pod.trader.cli", symbol, "--strategy", strategy, "--engine", engine]
    )

    time.sleep(3)  # give the trader time to import + connect before the publisher sends anything

    log.info("Starting replay publisher subprocess")
    replay_proc = subprocess.run(
        [PYTHON, "-m", "quant_pod.ingester.replay", symbol, "--speed", str(replay_speed), "--warmup", "2"],
    )

    trader_proc.wait()
    log.info("Pod run complete. Reports are in the reports/ directory.")


if __name__ == "__main__":
    app()
