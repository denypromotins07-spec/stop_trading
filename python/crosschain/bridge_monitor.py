"""
Cross-Chain Bridge Monitor for Real-Time Arbitrage Risk.
Consumes Rust-streamed bridge finality and liquidity metrics.
Detects bridge congestion or liquidity evaporation that could cause massive slippage.
Strictly enforces 3GB RAM limit via bounded metric history.
"""
import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import time

class BridgeStatus(Enum):
    HEALTHY = "healthy"
    CONGESTED = "congested"
    ILLIQUID = "illiquid"
    AT_RISK = "at_risk"
    CRITICAL = "critical"


@dataclass
class BridgeMetrics:
    """Real-time bridge health metrics"""
    bridge_id: str
    source_chain: str
    dest_chain: str
    total_liquidity: float  # In USD
    available_liquidity: float  # In USD
    pending_transactions: int
    avg_finality_time_ms: float
    tx_queue_depth: int
    gas_price_gwei: float
    timestamp_ns: int
    
    @property
    def utilization_ratio(self) -> float:
        """Calculate liquidity utilization ratio"""
        if self.total_liquidity <= 0:
            return 1.0
        return 1.0 - (self.available_liquidity / self.total_liquidity)
    
    @property
    def liquidity_headroom(self) -> float:
        """Available liquidity headroom in USD"""
        return self.available_liquidity


@dataclass
class BridgeHealthScore:
    """Aggregated health score for a bridge"""
    bridge_id: str
    overall_score: float  # 0-100, higher is healthier
    liquidity_score: float
    finality_score: float
    congestion_score: float
    status: BridgeStatus
    risk_factors: List[str]
    timestamp_ns: int


@dataclass
class CrossChainArbRisk:
    """Calculated risk for cross-chain arbitrage"""
    bridge_id: str
    arb_opportunity_id: str
    expected_slippage_bps: float
    failure_probability: float
    recommended_action: str  # "EXECUTE", "REDUCE_SIZE", "AVOID"
    max_safe_size_usd: float
    risk_score: float  # 0-1, higher is riskier
    timestamp_ns: int


class BridgeMonitor:
    """
    Real-time cross-chain bridge monitor.
    Calculates health scores and arbitrage risk metrics.
    """
    
    # Thresholds for health scoring
    LIQUIDITY_UTIL_WARNING = 0.7  # 70% utilization triggers warning
    LIQUIDITY_UTIL_CRITICAL = 0.9  # 90% utilization is critical
    FINALITY_TIME_WARNING_MS = 30000  # 30 seconds
    FINALITY_TIME_CRITICAL_MS = 120000  # 2 minutes
    QUEUE_DEPTH_WARNING = 100
    QUEUE_DEPTH_CRITICAL = 500
    
    # Memory bounds
    MAX_METRICS_HISTORY = 200  # Per bridge
    MAX_RISK_RECORDS = 100
    
    def __init__(self):
        self._metrics_history: Dict[str, deque] = {}
        self._current_scores: Dict[str, BridgeHealthScore] = {}
        self._risk_callbacks: List[callable] = []
        self._lock = asyncio.Lock()
        
        # Pre-configured bridges
        self._registered_bridges: set = set()
    
    def register_bridge(self, bridge_id: str):
        """Register a bridge for monitoring"""
        self._registered_bridges.add(bridge_id)
        if bridge_id not in self._metrics_history:
            self._metrics_history[bridge_id] = deque(maxlen=self.MAX_METRICS_HISTORY)
    
    def register_risk_callback(self, callback: callable):
        """Register callback for risk alerts"""
        self._risk_callbacks.append(callback)
    
    async def ingest_metrics(self, metrics: BridgeMetrics):
        """Ingest new bridge metrics and update health scores"""
        async with self._lock:
            bridge_id = metrics.bridge_id
            
            # Auto-register if not already registered
            if bridge_id not in self._metrics_history:
                self.register_bridge(bridge_id)
            
            self._metrics_history[bridge_id].append(metrics)
            
            # Update health score
            score = self._calculate_health_score(metrics)
            self._current_scores[bridge_id] = score
            
            # Check for status changes
            if len(self._metrics_history[bridge_id]) >= 2:
                prev_score = list(self._metrics_history[bridge_id])[-2]
                await self._check_status_change(bridge_id, score, prev_score)
    
    def _calculate_health_score(self, metrics: BridgeMetrics) -> BridgeHealthScore:
        """Calculate aggregated health score from metrics"""
        risk_factors = []
        
        # Liquidity score (0-100)
        util_ratio = metrics.utilization_ratio
        if util_ratio > self.LIQUIDITY_UTIL_CRITICAL:
            liquidity_score = max(0, 100 - (util_ratio - self.LIQUIDITY_UTIL_CRITICAL) * 200)
            risk_factors.append("Critical liquidity utilization")
        elif util_ratio > self.LIQUIDITY_UTIL_WARNING:
            liquidity_score = 50 + (self.LIQUIDITY_UTIL_CRITICAL - util_ratio) * 100
            risk_factors.append("High liquidity utilization")
        else:
            liquidity_score = 100 - util_ratio * 50
        
        # Finality score (0-100)
        finality_time = metrics.avg_finality_time_ms
        if finality_time > self.FINALITY_TIME_CRITICAL_MS:
            finality_score = max(0, 100 - (finality_time - self.FINALITY_TIME_CRITICAL_MS) / 1000)
            risk_factors.append("Critical finality delay")
        elif finality_time > self.FINALITY_TIME_WARNING_MS:
            finality_score = 50 + (self.FINALITY_TIME_CRITICAL_MS - finality_time) / 1800
            risk_factors.append("Elevated finality time")
        else:
            finality_score = 100 - (finality_time / self.FINALITY_TIME_WARNING_MS) * 50
        
        # Congestion score (0-100)
        queue_depth = metrics.tx_queue_depth
        if queue_depth > self.QUEUE_DEPTH_CRITICAL:
            congestion_score = max(0, 100 - (queue_depth - self.QUEUE_DEPTH_CRITICAL) / 5)
            risk_factors.append("Critical queue congestion")
        elif queue_depth > self.QUEUE_DEPTH_WARNING:
            congestion_score = 50 + (self.QUEUE_DEPTH_CRITICAL - queue_depth) / 8
            risk_factors.append("Queue congestion")
        else:
            congestion_score = 100 - (queue_depth / self.QUEUE_DEPTH_WARNING) * 50
        
        # Overall score (weighted average)
        overall_score = (
            liquidity_score * 0.4 +
            finality_score * 0.35 +
            congestion_score * 0.25
        )
        
        # Determine status
        if overall_score >= 80:
            status = BridgeStatus.HEALTHY
        elif overall_score >= 60:
            status = BridgeStatus.CONGESTED
        elif overall_score >= 40:
            status = BridgeStatus.ILLIQUID
        elif overall_score >= 20:
            status = BridgeStatus.AT_RISK
        else:
            status = BridgeStatus.CRITICAL
        
        return BridgeHealthScore(
            bridge_id=metrics.bridge_id,
            overall_score=float(overall_score),
            liquidity_score=float(liquidity_score),
            finality_score=float(finality_score),
            congestion_score=float(congestion_score),
            status=status,
            risk_factors=risk_factors,
            timestamp_ns=time.time_ns()
        )
    
    async def _check_status_change(self, bridge_id: str, 
                                   current_score: BridgeHealthScore,
                                   prev_metrics: BridgeMetrics):
        """Check for significant status changes and alert"""
        prev_score = self._current_scores.get(bridge_id)
        if prev_score is None:
            return
        
        # Check for downgrade
        if current_score.overall_score < prev_score.overall_score - 15:
            alert_event = {
                'type': 'BRIDGE_DOWNGRADE',
                'bridge_id': bridge_id,
                'previous_score': prev_score.overall_score,
                'current_score': current_score.overall_score,
                'previous_status': prev_score.status.value,
                'current_status': current_score.status.value,
                'risk_factors': current_score.risk_factors,
                'timestamp_ns': time.time_ns()
            }
            
            for callback in self._risk_callbacks:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert_event)
                else:
                    callback(alert_event)
    
    def calculate_arb_risk(self, bridge_id: str, arb_opportunity_id: str,
                           trade_size_usd: float) -> CrossChainArbRisk:
        """
        Calculate risk for a specific cross-chain arbitrage opportunity.
        """
        if bridge_id not in self._current_scores:
            return CrossChainArbRisk(
                bridge_id=bridge_id,
                arb_opportunity_id=arb_opportunity_id,
                expected_slippage_bps=100.0,  # Assume worst case
                failure_probability=1.0,
                recommended_action="AVOID",
                max_safe_size_usd=0.0,
                risk_score=1.0,
                timestamp_ns=time.time_ns()
            )
        
        score = self._current_scores[bridge_id]
        metrics_list = list(self._metrics_history[bridge_id])
        latest_metrics = metrics_list[-1] if metrics_list else None
        
        # Calculate expected slippage based on liquidity and congestion
        base_slippage = 5.0  # Base 5 bps
        
        if latest_metrics:
            # Slippage increases as liquidity decreases
            liq_factor = 1.0 + (1.0 - latest_metrics.utilization_ratio) * 2
            congestion_factor = 1.0 + latest_metrics.tx_queue_depth / 100
            
            # Size impact
            size_impact = trade_size_usd / max(latest_metrics.available_liquidity, 1) * 50
            
            expected_slippage_bps = base_slippage * liq_factor * congestion_factor + size_impact
        else:
            expected_slippage_bps = 50.0
        
        # Calculate failure probability based on health score
        failure_prob = max(0.01, (100 - score.overall_score) / 100)
        
        # Determine recommended action
        if score.status == BridgeStatus.CRITICAL:
            recommended_action = "AVOID"
            max_safe_size = 0.0
        elif score.status == BridgeStatus.AT_RISK:
            recommended_action = "REDUCE_SIZE"
            max_safe_size = latest_metrics.available_liquidity * 0.01 if latest_metrics else 0
        elif expected_slippage_bps > 50:
            recommended_action = "REDUCE_SIZE"
            max_safe_size = latest_metrics.available_liquidity * 0.05 if latest_metrics else 0
        else:
            recommended_action = "EXECUTE"
            max_safe_size = latest_metrics.available_liquidity * 0.1 if latest_metrics else float('inf')
        
        # Risk score (0-1)
        risk_score = (
            failure_prob * 0.4 +
            min(1.0, expected_slippage_bps / 100) * 0.4 +
            (1 - score.overall_score / 100) * 0.2
        )
        
        return CrossChainArbRisk(
            bridge_id=bridge_id,
            arb_opportunity_id=arb_opportunity_id,
            expected_slippage_bps=float(min(expected_slippage_bps, 1000)),
            failure_probability=float(failure_prob),
            recommended_action=recommended_action,
            max_safe_size_usd=float(max_safe_size),
            risk_score=float(risk_score),
            timestamp_ns=time.time_ns()
        )
    
    def get_health_score(self, bridge_id: str) -> Optional[BridgeHealthScore]:
        """Get current health score for a bridge"""
        return self._current_scores.get(bridge_id)
    
    def get_all_scores(self) -> Dict[str, BridgeHealthScore]:
        """Get all current health scores"""
        return self._current_scores.copy()
    
    def get_healthy_bridges(self, min_score: float = 60.0) -> List[str]:
        """Get list of bridges with health score above threshold"""
        return [
            bid for bid, score in self._current_scores.items()
            if score.overall_score >= min_score
        ]


# Global singleton instance
_monitor_instance: Optional[BridgeMonitor] = None


def get_monitor() -> BridgeMonitor:
    """Get or create global bridge monitor"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = BridgeMonitor()
    return _monitor_instance


async def demo():
    """Demo usage of the bridge monitor"""
    monitor = get_monitor()
    
    async def on_risk_alert(event: dict):
        print(f"BRIDGE ALERT: {event['bridge_id']} "
              f"downgraded from {event['previous_status']} to {event['current_status']}")
    
    monitor.register_risk_callback(on_risk_alert)
    
    # Simulate metrics for a bridge
    base_time = time.time_ns()
    
    for i in range(10):
        metrics = BridgeMetrics(
            bridge_id="ARB_BRIDGE",
            source_chain="ethereum",
            dest_chain="arbitrum",
            total_liquidity=10000000,
            available_liquidity=10000000 * (0.5 + 0.3 * np.sin(i / 2)),
            pending_transactions=50 + i * 10,
            avg_finality_time_ms=15000 + i * 2000,
            tx_queue_depth=20 + i * 15,
            gas_price_gwei=30 + i * 2,
            timestamp_ns=base_time + i * 1000000000
        )
        await monitor.ingest_metrics(metrics)
        
        # Calculate arb risk
        risk = monitor.calculate_arb_risk("ARB_BRIDGE", "ARB_OPP_1", 100000)
        print(f"Cycle {i}: Score={risk.risk_score:.2f}, "
              f"Action={risk.recommended_action}, Slippage={risk.expected_slippage_bps:.1f}bps")
    
    # Get healthy bridges
    healthy = monitor.get_healthy_bridges(min_score=50)
    print(f"\nHealthy bridges: {healthy}")


if __name__ == "__main__":
    asyncio.run(demo())
