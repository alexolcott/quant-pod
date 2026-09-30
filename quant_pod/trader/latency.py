"""Tick-to-fill latency tracking: the core "low latency" evidence for the pod.

Every order's creation timestamp is stamped with `time.perf_counter_ns()`
(common/events.Order.created_ns). When a Fill comes back, we record the delta
here and report percentiles at the end of a run.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LatencyTracker:
    label: str
    samples_ns: list[int] = field(default_factory=list)

    def record(self, latency_ns: int) -> None:
        self.samples_ns.append(latency_ns)

    def summary(self) -> dict:
        if not self.samples_ns:
            return {"label": self.label, "count": 0}
        arr = np.array(self.samples_ns, dtype=np.float64)
        return {
            "label": self.label,
            "count": len(arr),
            "mean_us": arr.mean() / 1_000,
            "p50_us": np.percentile(arr, 50) / 1_000,
            "p95_us": np.percentile(arr, 95) / 1_000,
            "p99_us": np.percentile(arr, 99) / 1_000,
            "max_us": arr.max() / 1_000,
        }

    def print_summary(self) -> None:
        s = self.summary()
        if s["count"] == 0:
            print(f"[{self.label}] no samples recorded")
            return
        print(
            f"[{s['label']}] n={s['count']} "
            f"mean={s['mean_us']:.1f}us p50={s['p50_us']:.1f}us "
            f"p95={s['p95_us']:.1f}us p99={s['p99_us']:.1f}us max={s['max_us']:.1f}us"
        )
