#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "orderbook.hpp"

namespace py = pybind11;

// Single-call, allocation-free fill-price calculation for the live trader's
// hot path. A first version routed every order through two OrderBook calls
// (seed synthetic liquidity, then match against it) -- that round-tripped
// the FFI boundary twice and paid for a std::map insert/erase and a
// std::vector<BookFill> allocation per order, which benchmarked *slower*
// than the equivalent pure-Python arithmetic. This is what actually gets
// used in the per-order hot path; OrderBook stays exposed below as a real,
// independently-tested matching engine for anything that needs genuine
// price-time-priority matching (resting limit orders, order-book depth).
double synthetic_fill_price(bool is_buy, double reference_price, double slippage_bps) {
    double direction = is_buy ? 1.0 : -1.0;
    return reference_price * (1.0 + direction * slippage_bps / 10000.0);
}

PYBIND11_MODULE(_hotpath, m) {
    m.doc() = "C++ low-latency order matching hot path for the quant pod trader";

    m.def("synthetic_fill_price", &synthetic_fill_price,
          py::arg("is_buy"), py::arg("reference_price"), py::arg("slippage_bps"));

    py::class_<BookFill>(m, "BookFill")
        .def_readonly("resting_order_id", &BookFill::resting_order_id)
        .def_readonly("incoming_order_id", &BookFill::incoming_order_id)
        .def_readonly("price", &BookFill::price)
        .def_readonly("quantity", &BookFill::quantity);

    py::class_<OrderBook>(m, "OrderBook")
        .def(py::init<>())
        .def("add_limit_order", &OrderBook::add_limit_order,
             py::arg("order_id"), py::arg("is_buy"), py::arg("price"), py::arg("quantity"))
        .def("add_market_order", &OrderBook::add_market_order,
             py::arg("order_id"), py::arg("is_buy"), py::arg("quantity"))
        .def("best_bid", &OrderBook::best_bid)
        .def("best_ask", &OrderBook::best_ask);
}
