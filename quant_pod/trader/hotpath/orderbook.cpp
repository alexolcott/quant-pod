#include "orderbook.hpp"

#include <algorithm>

std::vector<BookFill> OrderBook::match_against(
    long long incoming_order_id, bool incoming_is_buy, double limit_price, double quantity, bool has_limit) {
    std::vector<BookFill> fills;

    if (incoming_is_buy) {
        // Buyer crosses resting asks, cheapest first.
        auto it = asks_.begin();
        while (quantity > 0 && it != asks_.end()) {
            double level_price = it->first;
            if (has_limit && level_price > limit_price) break;

            auto& queue = it->second;
            while (quantity > 0 && !queue.empty()) {
                auto& resting = queue.front();
                double matched = std::min(quantity, resting.quantity);
                fills.push_back({resting.order_id, incoming_order_id, level_price, matched});
                quantity -= matched;
                resting.quantity -= matched;
                if (resting.quantity <= 0) queue.pop_front();
            }
            if (queue.empty()) {
                it = asks_.erase(it);
            } else {
                break;  // partially filled this level's front order, but ran out of incoming qty
            }
        }
    } else {
        // Seller crosses resting bids, highest first.
        auto it = bids_.begin();
        while (quantity > 0 && it != bids_.end()) {
            double level_price = it->first;
            if (has_limit && level_price < limit_price) break;

            auto& queue = it->second;
            while (quantity > 0 && !queue.empty()) {
                auto& resting = queue.front();
                double matched = std::min(quantity, resting.quantity);
                fills.push_back({resting.order_id, incoming_order_id, level_price, matched});
                quantity -= matched;
                resting.quantity -= matched;
                if (resting.quantity <= 0) queue.pop_front();
            }
            if (queue.empty()) {
                it = bids_.erase(it);
            } else {
                break;
            }
        }
    }

    return fills;
}

std::vector<BookFill> OrderBook::add_limit_order(long long order_id, bool is_buy, double price, double quantity) {
    auto fills = match_against(order_id, is_buy, price, quantity, /*has_limit=*/true);

    double filled = 0.0;
    for (const auto& f : fills) filled += f.quantity;
    double remaining = quantity - filled;

    if (remaining > 0) {
        RestingOrder resting{order_id, remaining, seq_counter_++};
        if (is_buy) {
            bids_[price].push_back(resting);
        } else {
            asks_[price].push_back(resting);
        }
    }

    return fills;
}

std::vector<BookFill> OrderBook::add_market_order(long long order_id, bool is_buy, double quantity) {
    return match_against(order_id, is_buy, /*limit_price=*/0.0, quantity, /*has_limit=*/false);
}

double OrderBook::best_bid() const {
    return bids_.empty() ? 0.0 : bids_.begin()->first;
}

double OrderBook::best_ask() const {
    return asks_.empty() ? 0.0 : asks_.begin()->first;
}
