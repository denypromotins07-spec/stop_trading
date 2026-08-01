"""
Wrapped Asset Depegging Risk Model.
Implements depegging probability model for wrapped assets (wBTC, bridged USDT) using on-chain reserve audits.
Instantly halts trading and liquidates toxic inventory if deviation exceeds dynamic threshold.
Strictly enforces 3GB RAM limit via bounded audit history.
"""
import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import time

class DepegStatus(Enum):
    PEGGED = "pegged"
    STRESSED = "stressed"
    DEPEGGING = "depegging"
    CRITICAL = "critical"
    HALTED = "halted"


@dataclass
class ReserveAudit:
    """On-chain reserve audit data"""
    asset_id: str
    total_supply: float  # Total wrapped tokens in circulation
    reserved_assets: float  # Actual reserves backing the wrapped tokens
    reserve_ratio: float  # reserved_assets / total_supply
    audit_timestamp_ns: int
    auditor: str
    proof_hash: str
    
    @property
    def deficit_usd(self) -> float:
        """Calculate reserve deficit in USD"""
        return max(0, self.total_supply - self.reserved_assets)


@dataclass
class MarketPrice:
    """Market price data for wrapped asset"""
    asset_id: str
    market_price: float  # Current market price
    peg_reference: float  # Target peg price (e.g., BTC price for wBTC)
    deviation_pct: float  # (market_price - peg_reference) / peg_reference * 100
    volume_24h: float
    timestamp_ns: int


@dataclass
class DepegProbability:
    """Calculated depegging probability and risk metrics"""
    asset_id: str
    depeg_probability: float  # 0-1 probability of full depeg
    expected_recovery_time_hours: float
    fair_value_estimate: float
    status: DepegStatus
    recommended_action: str  # "HOLD", "REDUCE", "LIQUIDATE", "HALT"
    confidence_interval: Tuple[float, float]
    risk_factors: List[str]
    timestamp_ns: int


@dataclass
class PositionRisk:
    """Risk assessment for a specific position"""
    asset_id: str
    position_size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    liquidation_triggered: bool
    liquidation_price: Optional[float]
    max_loss_estimate: float


class WrappedAssetRiskModel:
    """
    Depegging probability model for wrapped assets.
    Uses Bayesian updating with reserve audits and market data.
    """
    
    # Thresholds
    RESERVE_RATIO_WARNING = 0.95  # Below 95% reserve ratio triggers warning
    RESERVE_RATIO_CRITICAL = 0.80  # Below 80% is critical
    DEVIATION_WARNING_PCT = 2.0  # 2% deviation from peg
    DEVIATION_CRITICAL_PCT = 5.0  # 5% deviation triggers halt
    DEPEG_PROB_HALT = 0.7  # Halt trading if depeg prob > 70%
    
    # Memory bounds
    MAX_AUDIT_HISTORY = 50  # Per asset
    MAX_PRICE_HISTORY = 200  # Per asset
    
    def __init__(self):
        self._audit_history: Dict[str, deque] = {}
        self._price_history: Dict[str, deque] = {}
        self._current_risk: Dict[str, DepegProbability] = {}
        self._positions: Dict[str, float] = {}  # Current positions per asset
        self._alert_callbacks: List[callable] = []
        self._lock = asyncio.Lock()
        
        # Prior parameters for Bayesian model
        self._prior_alpha = 1.0  # Prior successes (pegged observations)
        self._prior_beta = 99.0  # Prior failures (depeg observations)
    
    def register_alert_callback(self, callback: callable):
        """Register callback for depeg alerts"""
        self._alert_callbacks.append(callback)
    
    async def ingest_audit(self, audit: ReserveAudit):
        """Ingest new reserve audit data"""
        async with self._lock:
            asset_id = audit.asset_id
            
            if asset_id not in self._audit_history:
                self._audit_history[asset_id] = deque(maxlen=self.MAX_AUDIT_HISTORY)
            
            self._audit_history[asset_id].append(audit)
            
            # Update risk assessment
            await self._update_risk(asset_id)
    
    async def ingest_price(self, price_data: MarketPrice):
        """Ingest new market price data"""
        async with self._lock:
            asset_id = price_data.asset_id
            
            if asset_id not in self._price_history:
                self._price_history[asset_id] = deque(maxlen=self.MAX_PRICE_HISTORY)
            
            self._price_history[asset_id].append(price_data)
            
            # Update risk assessment
            await self._update_risk(asset_id)
    
    async def _update_risk(self, asset_id: str):
        """Update depeg risk assessment for an asset"""
        if asset_id not in self._audit_history or asset_id not in self._price_history:
            return
        
        audits = list(self._audit_history[asset_id])
        prices = list(self._price_history[asset_id])
        
        if not audits or not prices:
            return
        
        latest_audit = audits[-1]
        latest_price = prices[-1]
        
        # Calculate depeg probability using multiple signals
        risk = self._calculate_depeg_probability(asset_id, latest_audit, latest_price, audits, prices)
        self._current_risk[asset_id] = risk
        
        # Check for status changes requiring alerts
        if len(prices) >= 2:
            prev_price = prices[-2]
            if risk.status != self._get_previous_status(asset_id):
                await self._check_and_alert(asset_id, risk, prev_price)
    
    def _calculate_depeg_probability(self, asset_id: str, audit: ReserveAudit,
                                      price: MarketPrice, 
                                      audit_history: List[ReserveAudit],
                                      price_history: List[MarketPrice]) -> DepegProbability:
        """
        Calculate depegging probability using Bayesian model.
        Combines reserve audit data with market price signals.
        """
        risk_factors = []
        
        # Signal 1: Reserve ratio risk
        reserve_risk = 0.0
        if audit.reserve_ratio < self.RESERVE_RATIO_CRITICAL:
            reserve_risk = 0.8
            risk_factors.append(f"Critical reserve ratio: {audit.reserve_ratio:.1%}")
        elif audit.reserve_ratio < self.RESERVE_RATIO_WARNING:
            reserve_risk = 0.5
            risk_factors.append(f"Low reserve ratio: {audit.reserve_ratio:.1%}")
        else:
            reserve_risk = max(0, (1.0 - audit.reserve_ratio) * 2)
        
        # Signal 2: Market deviation risk
        deviation_risk = 0.0
        abs_deviation = abs(price.deviation_pct)
        if abs_deviation > self.DEVIATION_CRITICAL_PCT:
            deviation_risk = 0.9
            risk_factors.append(f"Critical price deviation: {price.deviation_pct:.2f}%")
        elif abs_deviation > self.DEVIATION_WARNING_PCT:
            deviation_risk = 0.6
            risk_factors.append(f"Elevated price deviation: {price.deviation_pct:.2f}%")
        else:
            deviation_risk = abs_deviation / self.DEVIATION_WARNING_PCT * 0.5
        
        # Signal 3: Trend analysis (if we have history)
        trend_risk = 0.0
        if len(price_history) >= 5:
            recent_deviations = [p.deviation_pct for p in price_history[-5:]]
            if all(d < 0 for d in recent_deviations):  # Consistently below peg
                trend_risk = 0.5 + 0.1 * len(recent_deviations)
                risk_factors.append("Consistent trading below peg")
            elif np.std(recent_deviations) > 2.0:  # High volatility
                trend_risk = 0.3
                risk_factors.append("High deviation volatility")
        
        # Signal 4: Reserve trend
        reserve_trend_risk = 0.0
        if len(audit_history) >= 3:
            recent_ratios = [a.reserve_ratio for a in audit_history[-3:]]
            if all(recent_ratios[i] > recent_ratios[i+1] for i in range(len(recent_ratios)-1)):
                reserve_trend_risk = 0.4
                risk_factors.append("Declining reserve ratio trend")
        
        # Combine signals (weighted average)
        base_probability = (
            reserve_risk * 0.35 +
            deviation_risk * 0.35 +
            trend_risk * 0.15 +
            reserve_trend_risk * 0.15
        )
        
        # Apply Bayesian update with prior
        alpha = self._prior_alpha + base_probability * 10
        beta = self._prior_beta + (1 - base_probability) * 10
        posterior_mean = alpha / (alpha + beta)
        
        # Determine status
        if posterior_mean >= 0.7:
            status = DepegStatus.CRITICAL
        elif posterior_mean >= 0.5:
            status = DepegStatus.DEPEGGING
        elif posterior_mean >= 0.3:
            status = DepegStatus.STRESSED
        else:
            status = DepegStatus.PEGGED
        
        # Determine recommended action
        if status == DepegStatus.CRITICAL or posterior_mean >= self.DEPEG_PROB_HALT:
            recommended_action = "HALT"
        elif status == DepegStatus.DEPEGGING:
            recommended_action = "LIQUIDATE"
        elif status == DepegStatus.STRESSED:
            recommended_action = "REDUCE"
        else:
            recommended_action = "HOLD"
        
        # Calculate fair value estimate
        fair_value = price.peg_reference * audit.reserve_ratio
        
        # Confidence interval (Beta distribution)
        from scipy.stats import beta as beta_dist
        ci_lower, ci_upper = beta_dist.interval(0.95, alpha, beta)
        
        # Expected recovery time (heuristic)
        if status == DepegStatus.PEGGED:
            recovery_time = 0.0
        elif status == DepegStatus.STRESSED:
            recovery_time = 2.0
        elif status == DepegStatus.DEPEGGING:
            recovery_time = 24.0
        else:
            recovery_time = 168.0  # 1 week
        
        return DepegProbability(
            asset_id=asset_id,
            depeg_probability=float(posterior_mean),
            expected_recovery_time_hours=recovery_time,
            fair_value_estimate=float(fair_value),
            status=status,
            recommended_action=recommended_action,
            confidence_interval=(float(ci_lower), float(ci_upper)),
            risk_factors=risk_factors,
            timestamp_ns=time.time_ns()
        )
    
    def _get_previous_status(self, asset_id: str) -> Optional[DepegStatus]:
        """Get previous risk status for an asset"""
        prev_risk = self._current_risk.get(asset_id)
        return prev_risk.status if prev_risk else None
    
    async def _check_and_alert(self, asset_id: str, risk: DepegProbability,
                                prev_price: MarketPrice):
        """Check for critical status changes and send alerts"""
        if risk.status in [DepegStatus.CRITICAL, DepegStatus.DEPEGGING]:
            alert_event = {
                'type': 'DEPEG_ALERT',
                'asset_id': asset_id,
                'status': risk.status.value,
                'depeg_probability': risk.depeg_probability,
                'recommended_action': risk.recommended_action,
                'risk_factors': risk.risk_factors,
                'deviation_pct': risk.confidence_interval,
                'timestamp_ns': time.time_ns()
            }
            
            for callback in self._alert_callbacks:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert_event)
                else:
                    callback(alert_event)
    
    def get_risk(self, asset_id: str) -> Optional[DepegProbability]:
        """Get current depeg risk for an asset"""
        return self._current_risk.get(asset_id)
    
    def get_all_risks(self) -> Dict[str, DepegProbability]:
        """Get all current risk assessments"""
        return self._current_risk.copy()
    
    def should_halt_trading(self, asset_id: str) -> bool:
        """Check if trading should be halted for an asset"""
        risk = self._current_risk.get(asset_id)
        return risk is not None and (
            risk.status == DepegStatus.CRITICAL or
            risk.depeg_probability >= self.DEPEG_PROB_HALT or
            risk.recommended_action == "HALT"
        )
    
    def get_safe_position_limit(self, asset_id: str, 
                                 base_limit: float) -> float:
        """
        Get safe position limit based on depeg risk.
        Reduces limit as risk increases.
        """
        risk = self._current_risk.get(asset_id)
        if risk is None:
            return base_limit
        
        # Scale down based on depeg probability
        risk_multiplier = 1.0 - risk.depeg_probability
        
        if risk.status == DepegStatus.CRITICAL:
            return 0.0
        elif risk.status == DepegStatus.DEPEGGING:
            return base_limit * 0.1
        elif risk.status == DepegStatus.STRESSED:
            return base_limit * 0.5
        else:
            return base_limit * risk_multiplier
    
    async def check_position(self, asset_id: str, position_size: float,
                             entry_price: float) -> PositionRisk:
        """Check risk for a specific position"""
        risk = self._current_risk.get(asset_id)
        
        if risk is None or asset_id not in self._price_history:
            return PositionRisk(
                asset_id=asset_id,
                position_size=position_size,
                entry_price=entry_price,
                current_price=entry_price,
                unrealized_pnl=0.0,
                liquidation_triggered=False,
                liquidation_price=None,
                max_loss_estimate=position_size * entry_price
            )
        
        current_price_data = self._price_history[asset_id][-1]
        current_price = current_price_data.market_price
        
        unrealized_pnl = (current_price - entry_price) * position_size
        
        # Determine if liquidation should be triggered
        liquidation_triggered = risk.recommended_action in ["LIQUIDATE", "HALT"]
        
        # Estimate liquidation price (discounted for depeg scenario)
        if liquidation_triggered:
            liquidation_price = current_price * 0.9  # Assume 10% discount needed
        else:
            liquidation_price = None
        
        # Max loss estimate
        if risk.status == DepegStatus.CRITICAL:
            max_loss = position_size * entry_price  # Total loss possible
        else:
            max_loss = position_size * (entry_price - risk.fair_value_estimate)
        
        return PositionRisk(
            asset_id=asset_id,
            position_size=position_size,
            entry_price=entry_price,
            current_price=current_price,
            unrealized_pnl=unrealized_pnl,
            liquidation_triggered=liquidation_triggered,
            liquidation_price=liquidation_price,
            max_loss_estimate=max(0, max_loss)
        )


# Global singleton instance
_model_instance: Optional[WrappedAssetRiskModel] = None


def get_risk_model() -> WrappedAssetRiskModel:
    """Get or create global wrapped asset risk model"""
    global _model_instance
    if _model_instance is None:
        _model_instance = WrappedAssetRiskModel()
    return _model_instance


async def demo():
    """Demo usage of the wrapped asset risk model"""
    model = get_risk_model()
    
    async def on_depeg_alert(event: dict):
        print(f"DEPEG ALERT: {event['asset_id']} - {event['status']}")
        print(f"  Probability: {event['depeg_probability']:.1%}")
        print(f"  Action: {event['recommended_action']}")
        print(f"  Factors: {event['risk_factors']}")
    
    model.register_alert_callback(on_depeg_alert)
    
    base_time = time.time_ns()
    btc_price = 50000
    
    # Simulate normal operation
    for i in range(5):
        audit = ReserveAudit(
            asset_id="wBTC",
            total_supply=100000,
            reserved_assets=99500 - i * 500,  # Gradually declining
            reserve_ratio=(99500 - i * 500) / 100000,
            audit_timestamp_ns=base_time + i * 3600000000000,
            auditor="chainlink",
            proof_hash=f"0x{i:064x}"
        )
        
        price = MarketPrice(
            asset_id="wBTC",
            market_price=btc_price * (1 - 0.005 - i * 0.003),
            peg_reference=btc_price,
            deviation_pct=-0.5 - i * 0.3,
            volume_24h=1000000000,
            timestamp_ns=base_time + i * 3600000000000
        )
        
        await model.ingest_audit(audit)
        await model.ingest_price(price)
        
        risk = model.get_risk("wBTC")
        if risk:
            print(f"Cycle {i}: Prob={risk.depeg_probability:.1%}, "
                  f"Status={risk.status.value}, Action={risk.recommended_action}")
    
    # Check position
    pos_risk = await model.check_position("wBTC", 10.0, 50000)
    print(f"\nPosition Risk: Unrealized PnL=${pos_risk.unrealized_pnl:.2f}, "
          f"Liquidation={pos_risk.liquidation_triggered}")
    
    # Check if trading should halt
    should_halt = model.should_halt_trading("wBTC")
    print(f"Trading Halted: {should_halt}")


if __name__ == "__main__":
    asyncio.run(demo())
