"""Central paths and defaults shared by every module."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"

DATA_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# ZeroMQ PUB/SUB address the ingester's replay publisher binds to and the
# trader's engine connects to. tcp://127.0.0.1 keeps it local-machine only.
REPLAY_ZMQ_ADDR = "tcp://127.0.0.1:5556"
