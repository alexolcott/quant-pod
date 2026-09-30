"""Fetch historical bars for every current S&P 500 constituent.

Scrapes the ticker list from Wikipedia (no API key needed), normalizes
symbols for yfinance (BRK.B -> BRK-B), then loops the existing ingester
fetch logic over all of them with a small delay between requests to avoid
rate limiting. Continues past individual failures (delistings, renames,
transient rate limits) rather than aborting the whole batch -- with ~500
tickers hitting a free API, some failures are expected.

Usage:
    python scripts/fetch_sp500.py                        # all ~503 tickers, last 5 years
    python scripts/fetch_sp500.py --limit 10              # just the first 10, for a quick test
    python scripts/fetch_sp500.py --start 2015-01-01 --delay 1.0
"""
from __future__ import annotations

import io
import time
from datetime import date, datetime, timedelta

import pandas as pd
import requests
import typer

from quant_pod.common.logging import get_logger
from quant_pod.ingester import store
from quant_pod.ingester.sources.yfinance_source import YFinanceSource

log = get_logger("fetch_sp500")
app = typer.Typer(add_completion=False)

WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# Wikipedia blocks requests with no User-Agent at all; this just needs to look
# like a normal browser, nothing exotic.
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; quant-pod research script)"}


def normalize_ticker(symbol: str) -> str:
    """Wikipedia lists share classes with a dot (BRK.B); yfinance wants a dash (BRK-B)."""
    return symbol.strip().replace(".", "-")


def get_sp500_tickers() -> list[str]:
    resp = requests.get(WIKIPEDIA_URL, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    table = pd.read_html(io.StringIO(resp.text))[0]
    return sorted({normalize_ticker(s) for s in table["Symbol"]})


@app.command()
def main(
    start: str = typer.Option((date.today() - timedelta(days=365 * 5)).isoformat(), help="YYYY-MM-DD"),
    end: str = typer.Option(date.today().isoformat(), help="YYYY-MM-DD"),
    limit: int = typer.Option(0, help="Only fetch the first N tickers (0 = all ~503)"),
    delay: float = typer.Option(0.5, help="Seconds to sleep between requests, to avoid rate limiting"),
):
    tickers = get_sp500_tickers()
    if limit:
        tickers = tickers[:limit]

    log.info("Fetching %d tickers, %s -> %s (delay=%.1fs between requests)", len(tickers), start, end, delay)
    source = YFinanceSource()
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)

    succeeded: list[str] = []
    failed: list[str] = []

    for i, symbol in enumerate(tickers, 1):
        try:
            df = source.fetch(symbol, start_dt, end_dt)
            if df.empty:
                raise ValueError("no data returned")
            store.write_bars(symbol, df)
            succeeded.append(symbol)
            log.info("[%d/%d] %s: stored %d rows", i, len(tickers), symbol, len(df))
        except Exception as e:
            failed.append(symbol)
            log.warning("[%d/%d] %s: FAILED (%s)", i, len(tickers), symbol, e)
        time.sleep(delay)

    log.info("Done: %d succeeded, %d failed", len(succeeded), len(failed))
    if failed:
        log.info("Failed tickers: %s", ", ".join(failed))


if __name__ == "__main__":
    app()
