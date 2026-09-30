from datetime import datetime, timedelta, timezone

import pandas as pd

from quant_pod.common.events import Bar
from quant_pod.trader.bar_buffer import BarBuffer


def make_bar(i: int) -> Bar:
    return Bar(
        symbol="AAPL",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i),
        open=100 + i,
        high=101 + i,
        low=99 + i,
        close=100.5 + i,
        volume=1_000 + i,
    )


def test_empty_buffer_has_zero_length():
    assert len(BarBuffer()) == 0


def test_matches_naive_list_of_dicts_construction():
    bars = [make_bar(i) for i in range(10)]

    buf = BarBuffer(initial_capacity=4)  # force at least one resize
    for bar in bars:
        buf.append(bar)

    expected = pd.DataFrame(
        [{"open": b.open, "high": b.high, "low": b.low, "close": b.close, "volume": b.volume} for b in bars]
    )

    got = buf.as_dataframe()
    assert len(got) == 10
    for col in ("open", "high", "low", "close", "volume"):
        assert list(got[col]) == list(expected[col])


def test_grows_past_initial_capacity_without_losing_data():
    buf = BarBuffer(initial_capacity=2)
    for i in range(50):
        buf.append(make_bar(i))
    assert len(buf) == 50
    df = buf.as_dataframe()
    assert len(df) == 50
    assert df["close"].iloc[0] == 100.5
    assert df["close"].iloc[-1] == 100.5 + 49


def test_as_dataframe_reflects_only_appended_rows_not_full_capacity():
    buf = BarBuffer(initial_capacity=100)
    buf.append(make_bar(0))
    buf.append(make_bar(1))
    assert len(buf.as_dataframe()) == 2
