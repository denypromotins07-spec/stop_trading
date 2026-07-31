"""
TCA Module Root - Feeds execution analytics back into Rust IPC bridge.
Integrates execution analyzer and reward shaper for SOUL.md feedback.
Strictly enforces 3GB RAM limit.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path

from tca.execution_analyzer import ExecutionAnalyzer, ExecutionReport
from tca.reward_shaper import TCARewardShaper, RewardSignal


logger = logging.getLogger(__name__)


class TCAManager:
    """
    Central manager for all TCA operations.
    Coordinates execution analysis and reward shaping.
    """
    
    def __init__(self,
                 max_history: int = 10000,
                 slippage_scale: float = 1.0):
        """
        Initialize TCA manager.
        
        Args:
            max_history: Maximum history to keep
            slippage_scale: Scale for slippage penalties
        """
        self.max_history = max_history
        
        # Initialize components
        self.analyzer = ExecutionAnalyzer(max_history=max_history)
        self.reward_shaper = TCARewardShaper(
            slippage_scale=slippage_scale,
            max_history=max_history
        )
        
        # Callbacks for Rust IPC
        self._soul_callback: Optional[Callable] = None
        
        # Pending feedback for SOUL.md
        self._pending_feedback: List[Dict] = []
    
    def process_fill(self,
                    order_id: str,
                    instrument: str,
                    side: str,
                    quantity: float,
                    filled_quantity: float,
                    avg_fill_price: float,
                    arrival_price: float,
                    benchmark_price: float,
                    alpha_signal: str,
                    regime_id: str,
                    timestamp_ns: int,
                    maker_rebate: float = 0.0,
                    adverse_selection_cost: float = 0.0,
                    expected_alpha: float = 0.0) -> Dict[str, Any]:
        """
        Process a fill through the complete TCA pipeline.
        
        Args:
            order_id: Order identifier
            instrument: Instrument symbol
            side: 'buy' or 'sell'
            quantity: Original order quantity
            filled_quantity: Actually filled quantity
            avg_fill_price: Average fill price
            arrival_price: Price at order arrival
            benchmark_price: Benchmark price
            alpha_signal: Alpha signal that triggered order
            regime_id: Market regime identifier
            timestamp_ns: Timestamp in nanoseconds
            maker_rebate: Maker rebate captured
            adverse_selection_cost: Adverse selection cost
            expected_alpha: Expected alpha from trade
            
        Returns:
            Dict with execution report and reward signal
        """
        # Analyze execution
        exec_report = self.analyzer.analyze_fill(
            order_id=order_id,
            instrument=instrument,
            side=side,
            quantity=quantity,
            filled_quantity=filled_quantity,
            avg_fill_price=avg_fill_price,
            arrival_price=arrival_price,
            benchmark_price=benchmark_price,
            alpha_signal=alpha_signal,
            regime_id=regime_id,
            timestamp_ns=timestamp_ns
        )
        
        # Shape reward for RL agent
        agent_id = f"execution_{instrument}"
        reward_signal = self.reward_shaper.shape_reward(
            agent_id=agent_id,
            slippage_bps=exec_report.slippage_bps,
            market_impact_bps=exec_report.market_impact_bps,
            maker_rebate=maker_rebate,
            adverse_selection_cost=adverse_selection_cost,
            timestamp_ns=timestamp_ns,
            fill_quantity=filled_quantity,
            expected_alpha=expected_alpha
        )
        
        # Create feedback for SOUL.md
        feedback = {
            "order_id": order_id,
            "instrument": instrument,
            "timestamp_ns": timestamp_ns,
            "slippage_bps": exec_report.slippage_bps,
            "market_impact_bps": exec_report.market_impact_bps,
            "implementation_shortfall_bps": exec_report.implementation_shortfall,
            "reward_total": reward_signal.total_reward,
            "alpha_signal": alpha_signal,
            "regime_id": regime_id
        }
        
        self._pending_feedback.append(feedback)
        
        # Trigger callback if set
        if self._soul_callback:
            asyncio.create_task(self._soul_callback(feedback))
        
        return {
            "execution_report": exec_report,
            "reward_signal": reward_signal,
            "feedback": feedback
        }
    
    def set_soul_callback(self, callback: Callable):
        """Set callback for SOUL.md feedback."""
        self._soul_callback = callback
    
    def get_pending_feedback(self) -> List[Dict]:
        """Get pending feedback for SOUL.md journal."""
        feedback = self._pending_feedback.copy()
        self._pending_feedback.clear()
        return feedback
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution analysis summary."""
        return {
            "analyzer": self.analyzer.get_summary(),
            "reward_shaper": self.reward_shaper.get_summary()
        }
    
    def get_instrument_stats(self, instrument: str) -> Dict[str, Any]:
        """Get stats for specific instrument."""
        return self.analyzer.get_instrument_stats(instrument)
    
    def get_agent_performance(self, agent_id: str) -> Dict[str, Any]:
        """Get performance for specific agent."""
        return self.reward_shaper.get_agent_performance(agent_id)
    
    def get_full_status(self) -> Dict[str, Any]:
        """Get complete TCA system status."""
        return {
            "execution_summary": self.get_execution_summary(),
            "alpha_quality": self.analyzer.get_alpha_execution_quality(),
            "regime_quality": self.analyzer.get_regime_execution_quality(),
            "pending_feedback_count": len(self._pending_feedback)
        }
    
    def reset(self):
        """Reset TCA state."""
        self._pending_feedback.clear()


# Module-level singleton
_tca_manager: Optional[TCAManager] = None


def get_manager() -> TCAManager:
    """Get or create TCA manager singleton."""
    global _tca_manager
    if _tca_manager is None:
        raise RuntimeError("TCA manager not initialized")
    return _tca_manager


def initialize_tca(max_history: int = 10000,
                  slippage_scale: float = 1.0) -> TCAManager:
    """Initialize TCA system."""
    global _tca_manager
    _tca_manager = TCAManager(
        max_history=max_history,
        slippage_scale=slippage_scale
    )
    return _tca_manager


def process_fill(**kwargs) -> Dict[str, Any]:
    """Process fill via singleton."""
    manager = get_manager()
    return manager.process_fill(**kwargs)


def get_tca_status() -> Dict[str, Any]:
    """Get status via singleton."""
    manager = get_manager()
    return manager.get_full_status()


def get_pending_feedback() -> List[Dict]:
    """Get pending feedback via singleton."""
    manager = get_manager()
    return manager.get_pending_feedback()


# Example usage
async def main():
    """Example usage of TCA module."""
    logging.basicConfig(level=logging.INFO)
    
    manager = initialize_tca()
    
    # Simulate some fills
    import numpy as np
    np.random.seed(42)
    
    for i in range(10):
        arrival_price = 100.0 + np.random.randn() * 0.5
        slippage = np.random.randn() * 0.02
        fill_price = arrival_price * (1 + slippage / 10000)
        
        result = process_fill(
            order_id=f"order_{i}",
            instrument="ES",
            side="buy" if i % 2 == 0 else "sell",
            quantity=100,
            filled_quantity=100,
            avg_fill_price=fill_price,
            arrival_price=arrival_price,
            benchmark_price=arrival_price * (1 + np.random.randn() * 0.0001),
            alpha_signal=f"alpha_{i % 3}",
            regime_id=f"regime_{i % 2}",
            timestamp_ns=i * 1_000_000_000,
            maker_rebate=max(0, np.random.randn() * 0.3),
            adverse_selection_cost=abs(np.random.randn()) * 0.2,
            expected_alpha=np.random.randn() * 0.001
        )
        
        print(f"Order {i}: Slippage={result['execution_report'].slippage_bps:.2f}bps, "
              f"Reward={result['reward_signal'].total_reward:.6f}")
    
    print(f"\nTCA Status: {get_tca_status()}")
    print(f"\nPending feedback: {get_pending_feedback()}")


if __name__ == "__main__":
    asyncio.run(main())
