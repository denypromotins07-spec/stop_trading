"""
Portfolio Optimization Module Root.
Publishes optimal target weights to Nautilus portfolio manager via MessageBus.
Integrates HRP and Black-Litterman optimizers with Ray distributed math.
"""

import numpy as np
import ray
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json
import time

from .hrp_optimizer import HierarchicalRiskParity, compute_hrp_weights
from .black_litterman import (
    BlackLittermanOptimizer, 
    InvestorView, 
    parse_views_from_soul
)


@dataclass
class PortfolioConfig:
    """Configuration for portfolio optimization."""
    asset_ids: List[str] = field(default_factory=list)
    target_volatility: float = 0.15
    risk_aversion: float = 2.5
    bl_tau: float = 0.05
    hrp_workers: int = 4
    rebalance_threshold: float = 0.03
    max_weight: float = 0.5
    min_weight: float = -0.3
    blend_confidence: float = 0.7


@ray.remote(max_calls=500, memory=200 * 1024 * 1024)
class PortfolioOptimizationActor:
    """
    Ray actor for portfolio optimization with bounded memory.
    Coordinates HRP and Black-Litterman computations.
    """
    
    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.n_assets = len(config.asset_ids)
        
        # Initialize optimizers
        self.hrp_optimizer = HierarchicalRiskParity(
            config.asset_ids, 
            n_workers=config.hrp_workers
        )
        self.bl_optimizer = BlackLittermanOptimizer(
            config.asset_ids,
            risk_aversion=config.risk_aversion,
            tau=config.bl_tau
        )
        
        # State tracking
        self._last_optimization_time: float = 0
        self._last_weights: Dict[str, float] = {}
        self._optimization_count: int = 0
    
    def optimize_portfolio(self, returns: np.ndarray, market_caps: np.ndarray,
                           cov_matrix: np.ndarray, soul_data: Optional[Dict] = None) -> Dict[str, float]:
        """
        Run combined HRP + Black-Litterman optimization.
        
        Args:
            returns: Asset returns (n_assets x n_samples)
            market_caps: Market capitalizations
            cov_matrix: Covariance matrix
            soul_data: Optional ML ensemble data from SOUL.md
            
        Returns:
            Optimal weight dictionary
        """
        start_time = time.time()
        
        # Compute HRP weights
        hrp_weights = self.hrp_optimizer.optimize(
            returns, 
            target_vol=self.config.target_volatility
        )
        
        # Parse views from SOUL data if available
        views = []
        bl_confidence = self.config.blend_confidence
        
        if soul_data:
            views = parse_views_from_soul(soul_data)
            # Adjust confidence based on signal quality
            confidences = soul_data.get("confidence_scores", {})
            if confidences:
                bl_confidence = np.mean(list(confidences.values()))
        
        # Compute Black-Litterman weights
        bl_weights = self.bl_optimizer.optimize(
            market_caps,
            cov_matrix,
            views,
            target_vol=self.config.target_volatility
        )
        
        # Blend weights based on confidence
        final_weights = self.bl_optimizer.blend_with_hrp(
            hrp_weights, bl_weights, bl_confidence
        )
        
        # Apply weight constraints
        for asset_id in self.config.asset_ids:
            w = final_weights.get(asset_id, 0.0)
            final_weights[asset_id] = np.clip(
                w, 
                self.config.min_weight, 
                self.config.max_weight
            )
        
        # Renormalize
        total = sum(final_weights.values())
        if abs(total) > 1e-10:
            final_weights = {k: v / total for k, v in final_weights.items()}
        
        # Update state
        self._last_optimization_time = time.time()
        self._last_weights = final_weights
        self._optimization_count += 1
        
        return final_weights
    
    def get_nautilus_commands(self, portfolio_value: float) -> List[Dict]:
        """Generate Nautilus-compatible portfolio commands."""
        commands = []
        
        for asset_id, weight in self._last_weights.items():
            if abs(weight) > 1e-6:
                commands.append({
                    "type": "portfolio_allocation",
                    "instrument_id": asset_id,
                    "target_weight": float(weight),
                    "target_value": float(weight * portfolio_value),
                    "rebalance_threshold": self.config.rebalance_threshold,
                    "timestamp": int(time.time() * 1e9),
                    "source": "portfolio_optimizer"
                })
        
        return commands
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Return optimization statistics."""
        return {
            "optimization_count": self._optimization_count,
            "last_optimization_time": self._last_optimization_time,
            "assets": self.config.asset_ids,
            "target_volatility": self.config.target_volatility
        }
    
    def cleanup(self):
        """Clean up resources."""
        self.hrp_optimizer.cleanup()


class PortfolioManager:
    """
    Main portfolio management class coordinating optimization and Nautilus integration.
    Publishes weights to MessageBus for execution.
    """
    
    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.n_assets = len(config.asset_ids)
        
        # Initialize Ray actor
        self.actor = PortfolioOptimizationActor.remote(config)
        
        # MessageBus integration (placeholder for actual Nautilus integration)
        self._message_bus_queue: List[Dict] = []
        
        # Cache
        self._current_weights: Dict[str, float] = {}
        self._last_rebalance_time: float = 0
    
    def update_weights(self, returns: np.ndarray, market_caps: np.ndarray,
                       cov_matrix: np.ndarray, soul_data: Optional[Dict] = None) -> Dict[str, float]:
        """
        Update portfolio weights and publish to MessageBus.
        
        Args:
            returns: Asset returns
            market_caps: Market capitalizations
            cov_matrix: Covariance matrix
            soul_data: ML ensemble data
            
        Returns:
            Updated weights
        """
        # Run optimization on Ray actor
        future = self.actor.optimize_portfolio.remote(
            returns, market_caps, cov_matrix, soul_data
        )
        weights = ray.get(future)
        
        # Check if rebalancing is needed
        should_rebalance = self._check_rebalance_needed(weights)
        
        if should_rebalance:
            self._current_weights = weights
            self._last_rebalance_time = time.time()
            
            # Generate and queue Nautilus commands
            self._queue_nautilus_commands(weights)
        
        return weights
    
    def _check_rebalance_needed(self, new_weights: Dict[str, float]) -> bool:
        """Check if portfolio drift exceeds threshold."""
        if not self._current_weights:
            return True
        
        for asset_id in self.config.asset_ids:
            current = self._current_weights.get(asset_id, 0.0)
            new = new_weights.get(asset_id, 0.0)
            
            drift = abs(new - current)
            if drift > self.config.rebalance_threshold:
                return True
        
        return False
    
    def _queue_nautilus_commands(self, weights: Dict[str, float]):
        """Queue commands for Nautilus execution."""
        future = self.actor.get_nautilus_commands.remote(portfolio_value=100000)
        commands = ray.get(future)
        
        self._message_bus_queue.extend(commands)
    
    def get_pending_commands(self) -> List[Dict]:
        """Get and clear pending Nautilus commands."""
        commands = self._message_bus_queue.copy()
        self._message_bus_queue.clear()
        return commands
    
    def publish_to_messagebus(self, commands: List[Dict]):
        """
        Publish commands to Nautilus MessageBus.
        
        In production, this would integrate with the actual Nautilus MessageBus.
        """
        # Placeholder for actual MessageBus integration
        # Example: nautilus_core.publish("portfolio_commands", json.dumps(commands))
        for cmd in commands:
            print(f"[MessageBus] Publishing: {cmd}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get portfolio optimization statistics."""
        future = self.actor.get_optimization_stats.remote()
        stats = ray.get(future)
        
        stats["current_weights"] = self._current_weights
        stats["last_rebalance_time"] = self._last_rebalance_time
        stats["pending_commands"] = len(self._message_bus_queue)
        
        return stats
    
    def cleanup(self):
        """Clean up resources."""
        future = self.actor.cleanup.remote()
        ray.get(future)
        ray.kill(self.actor)


def create_portfolio_manager(asset_ids: List[str], 
                             target_vol: float = 0.15) -> PortfolioManager:
    """Factory function to create a configured portfolio manager."""
    config = PortfolioConfig(
        asset_ids=asset_ids,
        target_volatility=target_vol
    )
    return PortfolioManager(config)


if __name__ == "__main__":
    # Initialize Ray
    ray.init(
        num_cpus=4,
        _system_config={
            "max_bytes_spill": 0,
            "object_store_memory": 500 * 1024 * 1024
        }
    )
    
    # Example usage
    assets = ["BTC", "ETH", "SOL"]
    
    # Generate sample data
    np.random.seed(42)
    returns = np.random.randn(3, 1000) * 0.02
    market_caps = np.array([500, 250, 50])
    cov_matrix = np.cov(returns)
    
    # Sample SOUL data
    soul_data = {
        "alpha_signals": {"BTC": 0.3, "ETH": 0.5, "SOL": -0.2},
        "confidence_scores": {"BTC": 0.75, "ETH": 0.85, "SOL": 0.60}
    }
    
    # Create and run portfolio manager
    manager = create_portfolio_manager(assets, target_vol=0.15)
    
    weights = manager.update_weights(returns, market_caps, cov_matrix, soul_data)
    
    print("Optimized Weights:")
    for asset, weight in weights.items():
        print(f"  {asset}: {weight:.4f}")
    
    # Get and publish commands
    commands = manager.get_pending_commands()
    manager.publish_to_messagebus(commands)
    
    # Stats
    stats = manager.get_stats()
    print(f"\nStats: {json.dumps(stats, indent=2)}")
    
    manager.cleanup()
    ray.shutdown()
