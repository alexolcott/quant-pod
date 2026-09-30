"""Pre-trade risk checks. Runs in the hot path, so keep it cheap."""
from __future__ import annotations

from dataclasses import dataclass

from quant_pod.common.events import Order


@dataclass
class RiskLimits:
    max_position: float = 10_000.0
    max_order_notional: float = 1_000_000.0

    def check(self, order: Order, current_position: float, reference_price: float) -> tuple[bool, str]:
        direction = 1 if order.side.value == "BUY" else -1
        projected_position = current_position + direction * order.quantity

        if abs(projected_position) > self.max_position:
            return False, f"projected position {projected_position} exceeds max_position {self.max_position}"

        notional = order.quantity * reference_price
        if notional > self.max_order_notional:
            return False, f"order notional {notional:.2f} exceeds max_order_notional {self.max_order_notional}"

        return True, "ok"
