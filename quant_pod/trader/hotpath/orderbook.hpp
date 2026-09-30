#pragma once

#include <deque>
#include <map>
#include <vector>

struct BookFill {
    long long resting_order_id;
    long long incoming_order_id;
    double price;
    double quantity;
};

// Price-time priority limit order book. Bids are kept highest-price-first,
// asks lowest-price-first; within a price level, orders queue FIFO.
class OrderBook {
public:
    std::vector<BookFill> add_limit_order(long long order_id, bool is_buy, double price, double quantity);
    // Immediate-or-cancel: matches against resting liquidity, any unfilled
    // remainder is dropped rather than resting in the book.
    std::vector<BookFill> add_market_order(long long order_id, bool is_buy, double quantity);

    double best_bid() const;
    double best_ask() const;

private:
    struct RestingOrder {
        long long order_id;
        double quantity;
        long long seq;
    };

    std::map<double, std::deque<RestingOrder>, std::greater<double>> bids_;
    std::map<double, std::deque<RestingOrder>> asks_;
    long long seq_counter_ = 0;

    std::vector<BookFill> match_against(
        long long incoming_order_id, bool incoming_is_buy, double limit_price, double quantity, bool has_limit);
};
