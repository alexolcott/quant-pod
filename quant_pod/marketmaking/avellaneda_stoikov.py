"""Avellaneda-Stoikov (2008) optimal market-making quotes.

    r(s, q, t) = s - q * gamma * sigma^2 * (T - t)                     [reservation price]
    spread     = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / kappa)

`gamma` is the market maker's risk aversion, `sigma` the mid-price volatility,
and `kappa` the decay rate of the assumed order-arrival intensity
lambda(delta) = A * exp(-kappa * delta) for a counterparty hitting a quote
`delta` away from mid. `A` sets the overall arrival rate but, notably, drops
out of the optimal spread entirely -- it governs how often you get filled,
not how wide you should quote.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def reservation_price(mid: float, inventory: float, gamma: float, sigma: float, time_remaining: float) -> float:
    """The price at which the maker is indifferent to holding `inventory` --
    below mid when long (eager to sell), above mid when short (eager to buy).
    """
    return mid - inventory * gamma * sigma**2 * time_remaining


def optimal_spread(gamma: float, sigma: float, time_remaining: float, kappa: float) -> float:
    """Total bid-ask spread around the reservation price. Always positive, but
    *not* monotonic in `gamma`: the inventory-risk term grows with gamma while
    the (2/gamma)*ln(1+gamma/kappa) term shrinks, so which direction the total
    moves depends on which term dominates for a given sigma/time_remaining/kappa.
    """
    if gamma <= 0:
        # lim_{gamma->0} (2/gamma) * ln(1 + gamma/kappa) = 2/kappa (risk-neutral limit).
        return 2.0 / kappa
    return gamma * sigma**2 * time_remaining + (2.0 / gamma) * np.log(1.0 + gamma / kappa)


@dataclass
class AvellanedaStoikovQuoter:
    """Holds the model's calibrated parameters; quoting itself is stateless."""

    gamma: float
    kappa: float
    sigma: float

    def quote(self, mid: float, inventory: float, time_remaining: float) -> tuple[float, float]:
        """Returns (bid, ask), centered on the reservation price -- not on `mid`."""
        r = reservation_price(mid, inventory, self.gamma, self.sigma, time_remaining)
        spread = optimal_spread(self.gamma, self.sigma, time_remaining, self.kappa)
        return r - spread / 2, r + spread / 2


def suppress_quotes_crossing_mid(bid: float, ask: float, mid: float) -> tuple[float, float]:
    """Guards against the inventory skew overshooting far enough that a quote
    crosses to the wrong side of mid. The skew term in `reservation_price` is
    `inventory * gamma * sigma^2 * time_remaining` -- for a large enough
    inventory (a big `fill_size`), sigma, or time_remaining, that can exceed
    half the spread, pushing the ask *below* mid (or the bid *above* it).
    When that happens, the quoted price itself is already a realized loss
    relative to fair value the moment it's posted: e.g. buying 10 shares at
    $40 can skew the ask to $29 one tick later, selling that same inventory
    at a guaranteed ~$11/share loss, not genuine inventory-risk management.

    Rather than let the model quote a self-destructive price, the affected
    side is suppressed (NaN) -- the same response `max_inventory` uses for
    its own hard limit, just triggered by the quote itself crossing mid
    instead of a separate position threshold.
    """
    if not np.isnan(bid) and bid > mid:
        bid = np.nan
    if not np.isnan(ask) and ask < mid:
        ask = np.nan
    return bid, ask


def apply_tolerance_band(
    theoretical_bid: float,
    theoretical_ask: float,
    posted_bid: float | None,
    posted_ask: float | None,
    band: float,
) -> tuple[float, float]:
    """A tolerance-zone (deadband) requoting policy: given the fresh
    theoretical quote and whatever is currently posted, only move a side if
    the theoretical value has drifted more than `band` away from it --
    otherwise keep resting at the old price. This is what makes it
    hysteresis rather than a one-shot filter: the comparison point is the
    *last posted* price, not a fixed reference, so it only re-anchors when a
    move actually happens, and won't flicker from theoretical noise sitting
    near a static boundary.

    Real market makers do this because every requote is a cancel + new order
    -- it costs exchange fees, and constantly adjusting is itself a signal
    other participants can detect and trade against ("quote flicker").

    `posted_bid`/`posted_ask` is None on the first call (nothing posted yet,
    so the theoretical value is adopted immediately) and NaN whenever a side
    isn't quoting (e.g. pulled by a `max_inventory` limit elsewhere) -- in
    both cases, and whenever the theoretical value itself is NaN, the band
    comparison is skipped and the fresh value is adopted directly: a
    suppressed side shouldn't stick around stale once it starts quoting
    again, and a side that just got suppressed shouldn't stay at its last
    finite price.
    """

    def _update(theoretical: float, posted: float | None) -> float:
        if posted is None or np.isnan(posted) or np.isnan(theoretical):
            return theoretical
        if abs(theoretical - posted) > band:
            return theoretical
        return posted

    return _update(theoretical_bid, posted_bid), _update(theoretical_ask, posted_ask)
