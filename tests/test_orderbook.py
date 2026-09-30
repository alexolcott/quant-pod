import pytest

from quant_pod.trader.hotpath import OrderBook
from quant_pod.trader.py_orderbook import PyOrderBook

# Run every test against both the C++ and the pure-Python reference book to
# guarantee they implement identical price-time-priority semantics.
IMPLEMENTATIONS = [OrderBook, PyOrderBook]


@pytest.mark.parametrize("book_cls", IMPLEMENTATIONS)
def test_market_order_matches_best_price_first(book_cls):
    book = book_cls()
    book.add_limit_order(1, False, 101.0, 10)  # ask
    book.add_limit_order(2, False, 100.0, 10)  # better ask
    fills = book.add_market_order(3, True, 5)
    assert len(fills) == 1
    assert fills[0].price == 100.0
    assert fills[0].resting_order_id == 2


@pytest.mark.parametrize("book_cls", IMPLEMENTATIONS)
def test_fifo_within_same_price_level(book_cls):
    book = book_cls()
    book.add_limit_order(1, False, 100.0, 5)
    book.add_limit_order(2, False, 100.0, 5)
    fills = book.add_market_order(3, True, 7)
    assert [f.resting_order_id for f in fills] == [1, 2]
    assert fills[0].quantity == 5
    assert fills[1].quantity == 2


@pytest.mark.parametrize("book_cls", IMPLEMENTATIONS)
def test_market_order_partial_fill_when_book_thin(book_cls):
    book = book_cls()
    book.add_limit_order(1, False, 100.0, 5)
    fills = book.add_market_order(2, True, 20)
    total_filled = sum(f.quantity for f in fills)
    assert total_filled == 5  # remainder is dropped (IOC), not an error


@pytest.mark.parametrize("book_cls", IMPLEMENTATIONS)
def test_limit_order_rests_when_no_cross(book_cls):
    book = book_cls()
    book.add_limit_order(1, True, 99.0, 10)  # bid, nothing to match
    assert book.best_bid() == 99.0
    assert book.best_ask() == 0.0


@pytest.mark.parametrize("book_cls", IMPLEMENTATIONS)
def test_limit_buy_respects_its_own_limit_price(book_cls):
    book = book_cls()
    book.add_limit_order(1, False, 105.0, 10)  # ask at 105
    fills = book.add_limit_order(2, True, 100.0, 10)  # buy capped at 100, shouldn't cross
    assert fills == []
    assert book.best_bid() == 100.0
    assert book.best_ask() == 105.0


@pytest.mark.parametrize("book_cls", IMPLEMENTATIONS)
def test_best_bid_ask_after_partial_consumption(book_cls):
    book = book_cls()
    book.add_limit_order(1, False, 100.0, 10)
    book.add_limit_order(2, False, 101.0, 10)
    book.add_market_order(3, True, 10)  # fully consumes the 100.0 level
    assert book.best_ask() == 101.0
