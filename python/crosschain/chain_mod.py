"""
Cross-Chain Module Root.
Integrates bridge health scores into the pre-trade risk bus to block toxic cross-venue executions.
"""
from .bridge_monitor import (
    BridgeMonitor,
    BridgeMetrics,
    BridgeHealthScore,
    BridgeStatus,
    CrossChainArbRisk,
    get_monitor
)
from .wrapped_risk import (
    WrappedAssetRiskModel,
    ReserveAudit,
    MarketPrice,
    DepegProbability,
    DepegStatus,
    PositionRisk,
    get_risk_model
)

__all__ = [
    # Bridge Monitor
    "BridgeMonitor",
    "BridgeMetrics",
    "BridgeHealthScore",
    "BridgeStatus",
    "CrossChainArbRisk",
    "get_monitor",
    
    # Wrapped Asset Risk
    "WrappedAssetRiskModel",
    "ReserveAudit",
    "MarketPrice",
    "DepegProbability",
    "DepegStatus",
    "PositionRisk",
    "get_risk_model",
]
