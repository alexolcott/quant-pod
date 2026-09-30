"""`python -m quant_pod.ingester.cli fetch AAPL --start 2022-01-01 --end 2024-01-01`"""
from __future__ import annotations

from datetime import datetime

import typer

from quant_pod.common.logging import get_logger
from quant_pod.ingester import store
from quant_pod.ingester.sources.synthetic_source import SyntheticSource
from quant_pod.ingester.sources.yfinance_source import YFinanceSource

app = typer.Typer(add_completion=False)
log = get_logger("ingester")

SOURCES = {
    "yfinance": YFinanceSource(),
    "synthetic": SyntheticSource(),
}


@app.command()
def fetch(
    symbol: str,
    start: str = typer.Option(..., help="YYYY-MM-DD"),
    end: str = typer.Option(..., help="YYYY-MM-DD"),
    source: str = typer.Option("yfinance", help="yfinance | synthetic"),
):
    """Fetch historical bars for SYMBOL and store them to the local Parquet cache."""
    if source not in SOURCES:
        raise typer.BadParameter(f"source must be one of {list(SOURCES)}")

    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)

    log.info("Fetching %s %s -> %s from %s", symbol, start, end, source)
    df = SOURCES[source].fetch(symbol, start_dt, end_dt)
    if df.empty:
        log.warning("No data returned for %s", symbol)
        raise typer.Exit(code=1)

    total = store.write_bars(symbol, df)
    log.info("Stored %d new rows, %d total rows for %s", len(df), total, symbol)


@app.command("list")
def list_symbols():
    """List symbols currently cached in the local Parquet store."""
    from quant_pod.common.config import DATA_DIR

    symbols = sorted(p.stem for p in DATA_DIR.glob("*.parquet"))
    if not symbols:
        log.info("No symbols stored yet.")
        return
    for sym in symbols:
        df = store.read_bars(sym)
        log.info("%s: %d rows, %s -> %s", sym, len(df), df.index.min().date(), df.index.max().date())


if __name__ == "__main__":
    app()
