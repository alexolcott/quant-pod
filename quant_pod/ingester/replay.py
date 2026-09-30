"""Replays stored historical bars over a ZeroMQ PUB socket as if they were a
live market data feed. Runs as its own process so the trader consumes it the
same way it would consume a real feed.
"""
from __future__ import annotations

import time
from datetime import datetime

import typer
import zmq

from quant_pod.common.config import REPLAY_ZMQ_ADDR
from quant_pod.common.events import Bar
from quant_pod.common.logging import get_logger
from quant_pod.ingester import store

log = get_logger("replay")
END_OF_STREAM = b"__END__"


def run_replay(symbol: str, speed_seconds: float = 0.0, warmup_seconds: float = 1.0) -> None:
    """Publish stored bars for `symbol` one at a time.

    `speed_seconds` is the delay between bars (0 = as fast as possible).
    `warmup_seconds` is a pause after binding before publishing, which works
    around ZeroMQ PUB/SUB's classic "slow joiner" problem (a SUB that connects
    after the first messages were sent misses them, since PUB never buffers
    for not-yet-connected subscribers) -- fine for a local single-subscriber
    simulation, not something a production feed would rely on.
    """
    bars = store.read_bars(symbol)
    log.info("Loaded %d bars for %s", len(bars), symbol)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(REPLAY_ZMQ_ADDR)
    log.info("Publishing on %s (warmup %.1fs for subscribers to connect)", REPLAY_ZMQ_ADDR, warmup_seconds)
    time.sleep(warmup_seconds)

    topic = symbol.encode()
    for ts, row in bars.iterrows():
        bar = Bar(
            symbol=symbol,
            timestamp=ts.to_pydatetime(),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )
        sock.send_multipart([topic, bar.model_dump_json().encode()])
        if speed_seconds > 0:
            time.sleep(speed_seconds)

    sock.send_multipart([topic, END_OF_STREAM])
    log.info("Replay complete for %s, sent end-of-stream", symbol)

    time.sleep(0.2)  # let the final send flush before tearing down
    sock.close()
    ctx.term()


app = typer.Typer(add_completion=False)


@app.command()
def main(
    symbol: str,
    speed: float = typer.Option(0.0, help="Seconds to sleep between bars (0 = max speed)"),
    warmup: float = typer.Option(1.0, help="Seconds to wait after bind before publishing"),
):
    run_replay(symbol, speed_seconds=speed, warmup_seconds=warmup)


if __name__ == "__main__":
    app()
