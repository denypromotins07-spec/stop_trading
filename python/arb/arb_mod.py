"""
Arbitrage Module Root
Wires Python arbitrage orchestrator to Nautilus execution engine.
"""

from .spread_monitor import SpreadMonitor, SpreadSignal
from .kalman_pairs import KalmanPairsTrader, PairsState

__all__ = [
    "SpreadMonitor",
    "SpreadSignal",
    "KalmanPairsTrader",
    "PairsState",
]
