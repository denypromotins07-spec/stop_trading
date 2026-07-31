"""
Risk Management Module Root.
Aggregates ML risk metrics and pushes them to the global pre-trade risk bus via Rust IPC bridge.
Integrates VaR/CVaR prediction with drawdown monitoring for comprehensive risk management.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json
import time


from .var_cvar_ml import MLVaRCVaRPredictor
from .drawdown_predictor import DrawdownMonitor, DrawdownPredictor


@dataclass
class RiskConfig:
    """Configuration for risk management system."""
    asset_ids: List[str] = field(default_factory=list)
    strategy_ids: List[str] = field(default_factory=list)
    max_var_pct: float = 0.05
    max_cvar_pct: float = 0.08
    max_drawdown_pct: float = 0.10
    var_confidence: float = 0.99
    portfolio_value: float = 100000.0
    check_interval_ms: int = 1000


class RiskAggregator:
    """
    Central risk aggregator combining ML-based VaR/CVaR with drawdown predictions.
    Pushes risk metrics to Rust IPC bridge for pre-trade checks.
    """
    
    def __init__(self, config: RiskConfig):
        self.config = config
        
        # Initialize ML predictors
        self.var_cvar_predictor = MLVaRCVaRPredictor(
            asset_ids=config.asset_ids,
            n_actors=2
        )
        
        self.drawdown_monitor = DrawdownMonitor(
            strategies=config.strategy_ids,
            max_drawdown_threshold=config.max_drawdown_pct
        )
        
        # Risk state
        self._current_risk_metrics: Dict[str, Dict] = {}
        self._last_update_time: float = 0
        self._risk_breaches: List[Dict] = []
        
        # IPC bridge placeholder (would connect to Rust in production)
        self._ipc_queue: List[Dict] = []
    
    def update_market_data(self, market_data: Dict[str, Dict],
                           strategy_data: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Update risk models with latest market data.
        
        Args:
            market_data: Per-asset market data for VaR/CVaR prediction
            strategy_data: Per-strategy data for drawdown monitoring
            
        Returns:
            Aggregated risk metrics
        """
        # Update VaR/CVaR predictions
        var_cvar_preds = self.var_cvar_predictor.predict(market_data)
        
        # Update drawdown predictions
        dd_commands = self.drawdown_monitor.check_strategies(strategy_data)
        
        # Aggregate metrics
        risk_metrics = {
            "timestamp": int(time.time() * 1e9),
            "assets": {},
            "strategies": {},
            "breaches": [],
            "limits": {}
        }
        
        # Process asset-level risk
        for asset_id, preds in var_cvar_preds.items():
            var_99 = preds.get("var_99", 0.0)
            cvar_99 = preds.get("cvar_99", 0.0)
            
            # Check breaches
            if var_99 > self.config.max_var_pct:
                breach = {
                    "type": "var_breach",
                    "asset_id": asset_id,
                    "value": var_99,
                    "threshold": self.config.max_var_pct,
                    "severity": "warning" if var_99 < self.config.max_var_pct * 1.5 else "critical"
                }
                risk_metrics["breaches"].append(breach)
                self._risk_breaches.append(breach)
            
            if cvar_99 > self.config.max_cvar_pct:
                breach = {
                    "type": "cvar_breach",
                    "asset_id": asset_id,
                    "value": cvar_99,
                    "threshold": self.config.max_cvar_pct,
                    "severity": "warning" if cvar_99 < self.config.max_cvar_pct * 1.5 else "critical"
                }
                risk_metrics["breaches"].append(breach)
                self._risk_breaches.append(breach)
            
            risk_metrics["assets"][asset_id] = {
                "var_95": preds.get("var_95", 0.0),
                "var_99": var_99,
                "cvar_95": preds.get("cvar_95", 0.0),
                "cvar_99": cvar_99,
                "status": "breach" if any(b["asset_id"] == asset_id for b in risk_metrics["breaches"]) else "normal"
            }
        
        # Process strategy-level risk
        for strategy_id in self.config.strategy_ids:
            status = self.drawdown_monitor.get_risk_status(strategy_id)
            risk_metrics["strategies"][strategy_id] = status
            
            if status["breach_risk"] == "high":
                breach = {
                    "type": "drawdown_risk",
                    "strategy_id": strategy_id,
                    "value": status["current_drawdown"],
                    "threshold": self.config.max_drawdown_pct,
                    "severity": "critical"
                }
                risk_metrics["breaches"].append(breach)
                self._risk_breaches.append(breach)
        
        # Calculate position limits
        risk_metrics["limits"] = self.var_cvar_predictor.get_risk_limits(
            self.config.portfolio_value,
            confidence=self.config.var_confidence
        )
        
        self._current_risk_metrics = risk_metrics
        self._last_update_time = time.time()
        
        # Queue for IPC
        self._queue_for_ipc(risk_metrics)
        
        # Process deleveraging commands
        if dd_commands:
            risk_metrics["actions"] = dd_commands
        
        return risk_metrics
    
    def _queue_for_ipc(self, metrics: Dict):
        """Queue metrics for Rust IPC bridge transmission."""
        ipc_message = {
            "type": "risk_update",
            "payload": metrics,
            "priority": "high" if metrics["breaches"] else "normal"
        }
        self._ipc_queue.append(ipc_message)
        
        # Keep queue bounded
        if len(self._ipc_queue) > 100:
            self._ipc_queue = self._ipc_queue[-100:]
    
    def check_pre_trade(self, instrument_id: str, 
                        order_size: float,
                        side: str) -> Dict[str, Any]:
        """
        Pre-trade risk check for a potential order.
        
        Args:
            instrument_id: Asset identifier
            order_size: Proposed order size in base currency
            side: "buy" or "sell"
            
        Returns:
            Risk check result with approval status
        """
        result = {
            "approved": True,
            "instrument_id": instrument_id,
            "order_size": order_size,
            "side": side,
            "checks": [],
            "timestamp": int(time.time() * 1e9)
        }
        
        # Get current risk metrics for this asset
        asset_risk = self._current_risk_metrics.get("assets", {}).get(instrument_id, {})
        cvar_99 = asset_risk.get("cvar_99", 0.05)
        
        # Check 1: Position limit based on CVaR
        limit = self._current_risk_metrics.get("limits", {}).get(instrument_id, self.config.portfolio_value * 0.5)
        current_exposure = self._get_current_exposure(instrument_id)
        proposed_exposure = current_exposure + (order_size if side == "buy" else -order_size)
        
        if proposed_exposure > limit:
            result["approved"] = False
            result["checks"].append({
                "check": "position_limit",
                "passed": False,
                "reason": f"Exposure {proposed_exposure:.2f} exceeds limit {limit:.2f}"
            })
        else:
            result["checks"].append({
                "check": "position_limit",
                "passed": True
            })
        
        # Check 2: VaR impact
        var_impact = abs(order_size / self.config.portfolio_value) * cvar_99
        max_var_contribution = self.config.max_var_pct * 0.3  # 30% of total VaR budget
        
        if var_impact > max_var_contribution:
            result["approved"] = False
            result["checks"].append({
                "check": "var_impact",
                "passed": False,
                "reason": f"VaR impact {var_impact:.4f} exceeds budget {max_var_contribution:.4f}"
            })
        else:
            result["checks"].append({
                "check": "var_impact",
                "passed": True
            })
        
        # Check 3: Recent breaches
        recent_breaches = [
            b for b in self._risk_breaches[-10:]
            if b.get("asset_id") == instrument_id and b.get("severity") == "critical"
        ]
        
        if recent_breaches:
            result["approved"] = False
            result["checks"].append({
                "check": "recent_breaches",
                "passed": False,
                "reason": f"{len(recent_breaches)} critical breaches in recent history"
            })
        else:
            result["checks"].append({
                "check": "recent_breaches",
                "passed": True
            })
        
        return result
    
    def _get_current_exposure(self, instrument_id: str) -> float:
        """Get current exposure for an instrument (placeholder)."""
        # In production, query actual positions from portfolio state
        return 0.0
    
    def get_pending_ipc_messages(self) -> List[Dict]:
        """Get and clear pending IPC messages for Rust bridge."""
        messages = self._ipc_queue.copy()
        self._ipc_queue.clear()
        return messages
    
    def send_to_rust_bridge(self, messages: List[Dict]):
        """
        Send risk messages to Rust IPC bridge.
        
        In production, this would use the actual Rust IPC mechanism.
        """
        for msg in messages:
            # Placeholder for actual IPC: rust_bridge.send(json.dumps(msg))
            print(f"[IPC->Rust] {json.dumps(msg)[:200]}...")
    
    def get_risk_dashboard(self) -> Dict[str, Any]:
        """Generate comprehensive risk dashboard data."""
        return {
            "timestamp": int(time.time() * 1e9),
            "metrics": self._current_risk_metrics,
            "breach_count": len(self._risk_breaches),
            "last_update": self._last_update_time,
            "config": {
                "max_var_pct": self.config.max_var_pct,
                "max_cvar_pct": self.config.max_cvar_pct,
                "max_drawdown_pct": self.config.max_drawdown_pct
            }
        }
    
    def cleanup(self):
        """Clean up resources."""
        self.var_cvar_predictor.cleanup()


def create_risk_manager(asset_ids: List[str], 
                        strategy_ids: List[str],
                        portfolio_value: float = 100000.0) -> RiskAggregator:
    """Factory function to create a configured risk manager."""
    config = RiskConfig(
        asset_ids=asset_ids,
        strategy_ids=strategy_ids,
        portfolio_value=portfolio_value
    )
    return RiskAggregator(config)


if __name__ == "__main__":
    # Example usage
    assets = ["BTC", "ETH", "SOL"]
    strategies = ["stat_arb_01", "momentum_01"]
    
    risk_manager = create_risk_manager(assets, strategies, portfolio_value=100000)
    
    # Simulate market data
    np.random.seed(42)
    
    market_data = {}
    for asset in assets:
        returns = np.random.randn(30) * 0.02
        market_data[asset] = {
            "returns": returns,
            "volumes": np.random.lognormal(10, 0.5, 30),
            "spreads": np.random.exponential(5, 30),
            "atr": np.abs(returns) * 2
        }
    
    strategy_data = {}
    for strategy in strategies:
        returns = np.random.randn(60) * 0.01
        cumulative_pnl = np.cumsum(returns)
        strategy_data[strategy] = {
            "returns": returns,
            "cumulative_pnl": cumulative_pnl,
            "volatility": np.std(returns) * np.ones(60),
            "volumes": np.ones(60)
        }
    
    # Update risk metrics
    metrics = risk_manager.update_market_data(market_data, strategy_data)
    
    print("Risk Dashboard:")
    print(f"  Assets monitored: {len(metrics['assets'])}")
    print(f"  Strategies monitored: {len(metrics['strategies'])}")
    print(f"  Active breaches: {len(metrics['breaches'])}")
    
    for asset, data in metrics["assets"].items():
        print(f"\n{asset}:")
        print(f"  VaR 99%: {data['var_99']:.4f}")
        print(f"  CVaR 99%: {data['cvar_99']:.4f}")
        print(f"  Status: {data['status']}")
    
    # Pre-trade check example
    print("\nPre-trade Checks:")
    for asset in assets:
        check = risk_manager.check_pre_trade(asset, order_size=10000, side="buy")
        status = "✓" if check["approved"] else "✗"
        print(f"  {asset}: {status}")
    
    # Send to Rust bridge
    ipc_messages = risk_manager.get_pending_ipc_messages()
    risk_manager.send_to_rust_bridge(ipc_messages)
    
    risk_manager.cleanup()
