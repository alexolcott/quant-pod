import numpy as np
import pytest

from quant_pod.marketmaking.markout import compute_markouts, summarize_markouts
from quant_pod.marketmaking.simulator import Fill, MarketMakerResult


def _result(mid: list[float], fills: list[Fill]) -> MarketMakerResult:
    n = len(mid)
    zeros = np.zeros(n)
    return MarketMakerResult(
        mid=np.array(mid), bid=zeros, ask=zeros,
        inventory=zeros, cash=zeros, equity=zeros, fills=fills,
    )


def test_spread_capture_and_adverse_selection_sum_to_total():
    # Bought at 99 when mid was 100 (bought 1 below fair value -- spread capture +1),
    # then mid falls to 97 by horizon=2 (adverse selection -3 against the new long).
    mid = [100.0, 99.0, 97.0]
    result = _result(mid, [Fill(step=0, side="buy", price=99.0, quantity=1.0)])

    markouts = compute_markouts(result, horizons=[2])
    row = markouts.iloc[0]
    assert row["spread_capture_2"] == pytest.approx(1.0)
    assert row["adverse_selection_2"] == pytest.approx(-3.0)
    assert row["total_markout_2"] == pytest.approx(row["spread_capture_2"] + row["adverse_selection_2"])
    assert row["total_markout_2"] == pytest.approx(-2.0)


def test_sell_fill_signs_are_mirrored():
    # Sold at 101 when mid was 100 (spread capture +1). Adverse selection is
    # measured mid-to-mid (fair value at the fill vs. fair value at the
    # horizon), not from the execution price, so the +3 move from mid=100 to
    # mid=103 gives adverse_selection = -1 * 3 = -3, and total = spread
    # capture + adverse selection = 1 - 3 = -2 (equivalently -1 * (103 - 101)).
    mid = [100.0, 101.0, 103.0]
    result = _result(mid, [Fill(step=0, side="sell", price=101.0, quantity=1.0)])

    markouts = compute_markouts(result, horizons=[2])
    row = markouts.iloc[0]
    assert row["spread_capture_2"] == pytest.approx(1.0)
    assert row["adverse_selection_2"] == pytest.approx(-3.0)
    assert row["total_markout_2"] == pytest.approx(-2.0)


def test_horizon_past_end_of_simulation_is_nan():
    mid = [100.0, 99.0]
    result = _result(mid, [Fill(step=0, side="buy", price=99.0, quantity=1.0)])

    markouts = compute_markouts(result, horizons=[5])
    assert np.isnan(markouts.iloc[0]["total_markout_5"])


def test_summarize_markouts_averages_and_drops_nans():
    mid = [100.0, 99.0, 101.0]  # n = len(mid) - 1 = 2
    fills = [
        Fill(step=0, side="buy", price=99.0, quantity=1.0),   # later_step=1 <= 2: full horizon=1 available
        Fill(step=2, side="buy", price=100.5, quantity=1.0),  # later_step=3 > 2: horizon=1 runs past the end -> NaN
    ]
    result = _result(mid, fills)

    markouts = compute_markouts(result, horizons=[1])
    stats = summarize_markouts(markouts, horizons=[1])

    assert stats[1]["n"] == 1
    # Only the step=0 fill has a full horizon=1 available: total_markout = mid[1] - price = 99.0 - 99.0.
    assert stats[1]["avg_total_markout"] == pytest.approx(mid[1] - 99.0)
