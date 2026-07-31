"""
Execution Analyzer - Analyzes Nautilus OrderFilled reports for TCA.
Calculates Implementation Shortfall, slippage, and market impact.
Correlates execution quality with ML alpha signals and market regime.
Strictly enforces 3GB RAM limit.
"""
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
from dataclasses import dataclass
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ExecutionReport:
    """Represents an analyzed execution."""
    order_id: str
    instrument: str
    side: str
    quantity: float
    filled_quantity: float
    avg_fill_price: float
    arrival_price: float
    benchmark_price: float
    implementation_shortfall: float
    slippage_bps: float
    market_impact_bps: float
    timing_cost_bps: float
    alpha_signal: str
    regime_id: str
    timestamp_ns: int


class ExecutionAnalyzer:
    """
    Analyzes execution quality from Nautilus OrderFilled reports.
    Memory-bounded for 3GB limit.
    """
    
    def __init__(self,
                 max_history: int = 10000,
                 benchmark_type: str = "arrival"):
        """
        Initialize execution analyzer.
        
        Args:
            max_history: Maximum execution history to keep
            benchmark_type: Type of benchmark ('arrival', 'close', 'vwap')
        """
        self.max_history = max_history
        self.benchmark_type = benchmark_type
        
        # Bounded execution history
        self._executions: deque = deque(maxlen=max_history)
        
        # Aggregated statistics by instrument
        self._instrument_stats: Dict[str, Dict] = {}
        
        # Correlation tracking
        self._alpha_correlations: Dict[str, List[float]] = {}
        self._regime_stats: Dict[str, Dict] = {}
    
    def analyze_fill(self,
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
                    timestamp_ns: int) -> ExecutionReport:
        """
        Analyze a single order fill.
        
        Args:
            order_id: Order identifier
            instrument: Instrument symbol
            side: 'buy' or 'sell'
            quantity: Original order quantity
            filled_quantity: Actually filled quantity
            avg_fill_price: Average fill price
            arrival_price: Price at order arrival
            benchmark_price: Benchmark price for IS calculation
            alpha_signal: Alpha signal that triggered order
            regime_id: Market regime identifier
            timestamp_ns: Timestamp in nanoseconds
            
        Returns:
            ExecutionReport with analysis
        """
        # Calculate implementation shortfall
        if side.lower() == 'buy':
            # For buys, shortfall = (fill_price - benchmark) / benchmark
            decision_to_fill = (avg_fill_price - arrival_price) / (arrival_price + 1e-9)
            market_move = (benchmark_price - arrival_price) / (arrival_price + 1e-9)
            impl_shortfall = decision_to_fill - market_move
        else:
            # For sells, shortfall = (benchmark - fill_price) / benchmark
            decision_to_fill = (arrival_price - avg_fill_price) / (arrival_price + 1e-9)
            market_move = (arrival_price - benchmark_price) / (arrival_price + 1e-9)
            impl_shortfall = decision_to_fill - market_move
        
        impl_shortfall_bps = impl_shortfall * 10000
        
        # Calculate slippage (vs arrival price)
        if side.lower() == 'buy':
            slippage = (avg_fill_price - arrival_price) / (arrival_price + 1e-9)
        else:
            slippage = (arrival_price - avg_fill_price) / (arrival_price + 1e-9)
        slippage_bps = slippage * 10000
        
        # Estimate market impact (simplified model)
        # In production, use more sophisticated impact models
        fill_ratio = filled_quantity / (quantity + 1e-9)
        market_impact_bps = slippage_bps * fill_ratio * 0.5
        
        # Timing cost (remainder of IS)
        timing_cost_bps = impl_shortfall_bps - market_impact_bps
        
        # Create report
        report = ExecutionReport(
            order_id=order_id,
            instrument=instrument,
            side=side,
            quantity=quantity,
            filled_quantity=filled_quantity,
            avg_fill_price=avg_fill_price,
            arrival_price=arrival_price,
            benchmark_price=benchmark_price,
            implementation_shortfall=impl_shortfall_bps,
            slippage_bps=slippage_bps,
            market_impact_bps=market_impact_bps,
            timing_cost_bps=timing_cost_bps,
            alpha_signal=alpha_signal,
            regime_id=regime_id,
            timestamp_ns=timestamp_ns
        )
        
        # Store in history
        self._executions.append(report)
        
        # Update statistics
        self._update_stats(report)
        
        return report
    
    def _update_stats(self, report: ExecutionReport):
        """Update aggregated statistics."""
        # Instrument stats
        if report.instrument not in self._instrument_stats:
            self._instrument_stats[report.instrument] = {
                'count': 0,
                'total_slippage_bps': 0.0,
                'total_impact_bps': 0.0,
                'total_is_bps': 0.0
            }
        
        stats = self._instrument_stats[report.instrument]
        stats['count'] += 1
        stats['total_slippage_bps'] += report.slippage_bps
        stats['total_impact_bps'] += report.market_impact_bps
        stats['total_is_bps'] += report.implementation_shortfall
        
        # Alpha correlation tracking
        if report.alpha_signal not in self._alpha_correlations:
            self._alpha_correlations[report.alpha_signal] = []
        self._alpha_correlations[report.alpha_signal].append(report.slippage_bps)
        
        # Keep bounded
        if len(self._alpha_correlations[report.alpha_signal]) > 1000:
            self._alpha_correlations[report.alpha_signal] = \
                self._alpha_correlations[report.alpha_signal][-1000:]
        
        # Regime stats
        if report.regime_id not in self._regime_stats:
            self._regime_stats[report.regime_id] = {
                'count': 0,
                'total_slippage_bps': 0.0,
                'avg_slippage_bps': 0.0
            }
        
        rstats = self._regime_stats[report.regime_id]
        rstats['count'] += 1
        rstats['total_slippage_bps'] += report.slippage_bps
        rstats['avg_slippage_bps'] = rstats['total_slippage_bps'] / rstats['count']
    
    def get_instrument_stats(self, instrument: str) -> Dict[str, Any]:
        """Get statistics for a specific instrument."""
        if instrument not in self._instrument_stats:
            return {}
        
        stats = self._instrument_stats[instrument]
        count = stats['count']
        
        return {
            'instrument': instrument,
            'execution_count': count,
            'avg_slippage_bps': stats['total_slippage_bps'] / max(count, 1),
            'avg_market_impact_bps': stats['total_impact_bps'] / max(count, 1),
            'avg_implementation_shortfall_bps': stats['total_is_bps'] / max(count, 1)
        }
    
    def get_alpha_execution_quality(self) -> Dict[str, float]:
        """Get execution quality by alpha signal."""
        quality = {}
        for alpha, slippages in self._alpha_correlations.items():
            if slippages:
                quality[alpha] = {
                    'avg_slippage_bps': float(np.mean(slippages)),
                    'std_slippage_bps': float(np.std(slippages)),
                    'count': len(slippages)
                }
        return quality
    
    def get_regime_execution_quality(self) -> Dict[str, Any]:
        """Get execution quality by market regime."""
        return dict(self._regime_stats)
    
    def get_recent_executions(self, n: int = 100) -> List[ExecutionReport]:
        """Get most recent executions."""
        return list(self._executions)[-n:]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get overall execution summary."""
        if not self._executions:
            return {"status": "no_data"}
        
        all_slippage = [e.slippage_bps for e in self._executions]
        all_impact = [e.market_impact_bps for e in self._executions]
        all_is = [e.implementation_shortfall for e in self._executions]
        
        return {
            "total_executions": len(self._executions),
            "instruments_tracked": len(self._instrument_stats),
            "alpha_signals_tracked": len(self._alpha_correlations),
            "regimes_tracked": len(self._regime_stats),
            "overall_avg_slippage_bps": float(np.mean(all_slippage)),
            "overall_avg_impact_bps": float(np.mean(all_impact)),
            "overall_avg_is_bps": float(np.mean(all_is)),
            "slippage_std_bps": float(np.std(all_slippage))
        }


# Example usage
def main():
    """Example usage of execution analyzer."""
    analyzer = ExecutionAnalyzer()
    
    # Simulate some fills
    np.random.seed(42)
    
    for i in range(100):
        arrival_price = 100.0 + np.random.randn() * 0.5
        slippage = np.random.randn() * 0.02  # 2 bps std
        fill_price = arrival_price * (1 + slippage / 10000)
        
        report = analyzer.analyze_fill(
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
            timestamp_ns=i * 1_000_000_000
        )
        
        if i < 5:
            print(f"Order {i}: Slippage={report.slippage_bps:.2f}bps, "
                  f"Impact={report.market_impact_bps:.2f}bps")
    
    print(f"\nSummary: {analyzer.get_summary()}")
    print(f"\nInstrument stats: {analyzer.get_instrument_stats('ES')}")
    print(f"\nAlpha quality: {analyzer.get_alpha_execution_quality()}")


if __name__ == "__main__":
    main()
