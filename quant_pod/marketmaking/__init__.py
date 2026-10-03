"""Avellaneda-Stoikov market-making research: a synthetic mid-price/order-flow
model (`order_flow.py`), the optimal quoting formulas (`avellaneda_stoikov.py`),
and the simulation loop that ties them together (`simulator.py`).

Separate from `quant_pod.research`/`quant_pod.trader`: those are built around
bar-level `Strategy.generate_target_positions`, which has no notion of two-sided
quoting, inventory skew, or fill probability -- this package models that
directly instead of forcing it through the bar-strategy abstraction.
"""
