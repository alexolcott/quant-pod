import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fetch_sp500 import normalize_ticker  # noqa: E402


def test_normalize_ticker_converts_dot_to_dash():
    assert normalize_ticker("BRK.B") == "BRK-B"
    assert normalize_ticker("BF.B") == "BF-B"


def test_normalize_ticker_leaves_plain_tickers_unchanged():
    assert normalize_ticker("AAPL") == "AAPL"


def test_normalize_ticker_strips_whitespace():
    assert normalize_ticker(" MSFT \n") == "MSFT"
