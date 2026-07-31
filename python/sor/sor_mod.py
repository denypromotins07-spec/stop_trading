"""
Smart Order Routing (SOR) Module Root.
Integrates bandit and forecaster outputs into Nautilus execution client routing logic.
Coordinates venue selection with liquidity timing for optimal execution.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json
import time


from .venue_predictor import VenuePredictor, ContextualBandit
from .liquidity_forecaster import LiquidityMonitor, LiquidityForecaster


@dataclass
class SORConfig:
    """Configuration for Smart Order Router."""
    instruments: List[str] = field(default_factory=list)
    venues: List[Dict] = field(default_factory=list)
    default_urgency: float = 0.5
    enable_liquidity_timing: bool = True
    min_order_size: float = 1000
    max_slices_per_order: int = 10
    split_threshold: float = 0.1  # Split if order > 10% of avg liquidity


@dataclass
class RoutingDecision:
    """Complete SOR routing decision."""
    instrument_id: str
    side: str
    total_quantity: float
    venue_allocations: List[Dict]
    timing_recommendation: Dict
    expected_total_cost_bps: float
    confidence_score: float
    metadata: Dict = field(default_factory=dict)


class SmartOrderRouter:
    """
    Main SOR engine combining venue prediction with liquidity forecasting.
    Generates optimal multi-venue execution plans for Nautilus.
    """
    
    def __init__(self, config: SORConfig):
        self.config = config
        
        # Initialize components
        self.venue_predictor = VenuePredictor(
            venues=config.venues,
            instruments=config.instruments
        )
        
        self.liquidity_monitor = LiquidityMonitor(
            instruments=config.instruments
        )
        
        # Active orders tracking
        self._active_orders: Dict[str, Dict] = {}
        self._order_counter: int = 0
        
        # Performance tracking
        self._routing_history: List[Dict] = []
        self._performance_by_venue: Dict[str, List[float]] = {
            v["id"]: [] for v in config.venues
        }
    
    def update_venue_data(self, venue_id: str, data: Dict):
        """Update real-time venue data."""
        self.venue_predictor.update_venue_state(venue_id, data)
    
    def update_liquidity_data(self, instrument_id: str, bid_depth: float,
                               ask_depth: float, spread_bps: float,
                               volume: float):
        """Update liquidity data for an instrument."""
        self.liquidity_monitor.update_liquidity(
            instrument_id, bid_depth, ask_depth, spread_bps, volume
        )
    
    def create_routing_plan(self, instrument_id: str, side: str,
                            total_quantity: float,
                            urgency: float = None) -> RoutingDecision:
        """
        Create optimal multi-venue routing plan.
        
        Args:
            instrument_id: Asset identifier
            side: "buy" or "sell"
            total_quantity: Total order quantity
            urgency: Execution urgency (0-1)
            
        Returns:
            Complete routing decision
        """
        urgency = urgency if urgency is not None else self.config.default_urgency
        
        # Check liquidity timing
        timing_rec = {"should_delay": False, "recommended_delay_seconds": 0}
        if self.config.enable_liquidity_timing:
            timing_rec = self.liquidity_monitor.get_execution_timing(
                instrument_id, total_quantity, urgency
            )["timing_recommendation"]
        
        # Determine if order should be split across venues
        venue_allocations = self._allocate_across_venues(
            instrument_id, side, total_quantity, urgency
        )
        
        # Calculate expected costs
        expected_costs = [
            alloc.get("expected_cost_bps", 10) * alloc.get("allocation_pct", 0)
            for alloc in venue_allocations
        ]
        expected_total_cost = sum(expected_costs)
        
        # Calculate confidence score
        confidences = [
            alloc.get("selection_confidence", 0.5)
            for alloc in venue_allocations
        ]
        confidence_score = np.mean(confidences) if confidences else 0.5
        
        # Create routing decision
        self._order_counter += 1
        order_id = f"sor_{self._order_counter}"
        
        decision = RoutingDecision(
            instrument_id=instrument_id,
            side=side,
            total_quantity=total_quantity,
            venue_allocations=venue_allocations,
            timing_recommendation=timing_rec,
            expected_total_cost_bps=expected_total_cost,
            confidence_score=confidence_score,
            metadata={
                "order_id": order_id,
                "n_venues": len(venue_allocations),
                "created_at": int(time.time() * 1e9),
                "urgency": urgency
            }
        )
        
        # Track active order
        self._active_orders[order_id] = {
            "decision": decision,
            "remaining_quantity": total_quantity,
            "status": "pending"
        }
        
        return decision
    
    def _allocate_across_venues(self, instrument_id: str, side: str,
                                 total_quantity: float,
                                 urgency: float) -> List[Dict]:
        """Allocate order across multiple venues."""
        allocations = []
        remaining_qty = total_quantity
        
        # Get best venue predictions
        while remaining_qty > 0:
            # Predict best venue for remaining quantity
            venue_result = self.venue_predictor.predict_best_venue(
                instrument_id, side, remaining_qty
            )
            
            if "error" in venue_result:
                break
            
            selected_venue = venue_result["selected_venue"]
            venue_info = next(
                (v for v in self.config.venues if v["id"] == selected_venue),
                {}
            )
            
            # Determine allocation size
            venue_liquidity = venue_result.get("expected_fill_rate", 0.8) * 1e7  # Estimate
            max_allocation = venue_liquidity * self.config.split_threshold
            allocation_qty = min(remaining_qty, max_allocation, total_quantity / 3)
            
            allocation = {
                "venue_id": selected_venue,
                "quantity": float(allocation_qty),
                "allocation_pct": float(allocation_qty / total_quantity),
                "expected_fill_rate": venue_result.get("expected_fill_rate", 0.8),
                "expected_cost_bps": venue_result.get("expected_cost_bps", 10),
                "selection_confidence": venue_result.get("selection_confidence", 0.5),
                "priority": len(allocations) + 1,
                "alternative_venues": venue_result.get("alternative_venues", [])
            }
            
            allocations.append(allocation)
            remaining_qty -= allocation_qty
            
            # Limit number of venues
            if len(allocations) >= self.config.max_slices_per_order:
                # Put remaining on last venue
                if allocations:
                    allocations[-1]["quantity"] += remaining_qty
                    allocations[-1]["allocation_pct"] = 1 - sum(
                        a["allocation_pct"] for a in allocations[:-1]
                    )
                break
        
        # Normalize allocations
        total_allocated = sum(a["quantity"] for a in allocations)
        if total_allocated > 0:
            for alloc in allocations:
                alloc["allocation_pct"] = alloc["quantity"] / total_allocated
        
        return allocations
    
    def submit_routing_plan(self, decision: RoutingDecision) -> str:
        """Submit routing plan for execution."""
        order_id = decision.metadata.get("order_id", "unknown")
        
        if order_id in self._active_orders:
            self._active_orders[order_id]["status"] = "submitted"
        
        return order_id
    
    def record_fill(self, order_id: str, venue_id: str,
                    filled_quantity: float, cost_bps: float):
        """Record fill from a venue allocation."""
        if order_id not in self._active_orders:
            return
        
        order = self._active_orders[order_id]
        order["remaining_quantity"] -= filled_quantity
        
        # Update performance tracking
        self._performance_by_venue.setdefault(venue_id, []).append(cost_bps)
        
        # Keep bounded
        if len(self._performance_by_venue[venue_id]) > 500:
            self._performance_by_venue[venue_id] = self._performance_by_venue[venue_id][-500:]
        
        # Record outcome for learning
        self.venue_predictor.record_execution_outcome(
            instrument_id=order["decision"].instrument_id,
            venue_id=venue_id,
            order_size=filled_quantity,
            filled=True,
            fill_rate=1.0,
            cost_bps=cost_bps
        )
        
        # Check if complete
        if order["remaining_quantity"] <= 0:
            order["status"] = "completed"
            self._routing_history.append({
                "order_id": order_id,
                "total_cost_bps": order["decision"].expected_total_cost_bps,
                "n_venues_used": len(order["decision"].venue_allocations),
                "timestamp": int(time.time() * 1e9)
            })
    
    def get_sor_analytics(self) -> Dict:
        """Get comprehensive SOR analytics."""
        analytics = {
            "total_orders": len(self._routing_history),
            "active_orders": len([o for o in self._active_orders.values() 
                                  if o["status"] == "submitted"]),
            "venue_performance": {},
            "routing_history_summary": {}
        }
        
        # Per-venue performance
        for venue_id, costs in self._performance_by_venue.items():
            if costs:
                analytics["venue_performance"][venue_id] = {
                    "avg_cost_bps": float(np.mean(costs)),
                    "std_cost_bps": float(np.std(costs)),
                    "n_executions": len(costs)
                }
        
        # Recent routing history
        if self._routing_history:
            recent = self._routing_history[-20:]
            analytics["routing_history_summary"] = {
                "recent_avg_cost_bps": float(np.mean([r["total_cost_bps"] for r in recent])),
                "recent_avg_venues": float(np.mean([r["n_venues_used"] for r in recent]))
            }
        
        return analytics
    
    def get_nautilus_commands(self, decision: RoutingDecision) -> List[Dict]:
        """Convert routing decision to Nautilus execution commands."""
        commands = []
        
        base_time = int(time.time() * 1e9)
        
        for i, alloc in enumerate(decision.venue_allocations):
            cmd = {
                "type": "sor_child_order",
                "parent_order_id": decision.metadata.get("order_id"),
                "instrument_id": decision.instrument_id,
                "side": decision.side,
                "quantity": alloc["quantity"],
                "venue_id": alloc["venue_id"],
                "priority": alloc["priority"],
                "expected_cost_bps": alloc["expected_cost_bps"],
                "metadata": {
                    "allocation_pct": alloc["allocation_pct"],
                    "expected_fill_rate": alloc["expected_fill_rate"],
                    "scheduled_at": base_time + i * 1000000  # 1ms apart
                }
            }
            commands.append(cmd)
        
        return commands


def create_smart_order_router(instruments: List[str],
                               venues: List[Dict]) -> SmartOrderRouter:
    """Factory function to create configured SOR."""
    config = SORConfig(
        instruments=instruments,
        venues=venues
    )
    return SmartOrderRouter(config)


if __name__ == "__main__":
    # Example usage
    instruments = ["BTC", "ETH", "SOL"]
    
    venues = [
        {"id": "binance", "maker_fee_bps": 10, "taker_fee_bps": 10},
        {"id": "coinbase", "maker_fee_bps": 5, "taker_fee_bps": 15},
        {"id": "kraken", "maker_fee_bps": 8, "taker_fee_bps": 12},
        {"id": "okx", "maker_fee_bps": 8, "taker_fee_bps": 10}
    ]
    
    sor = create_smart_order_router(instruments, venues)
    
    # Simulate market data
    np.random.seed(42)
    
    for venue in venues:
        sor.update_venue_data(venue["id"], {
            "latency_ms": np.random.uniform(20, 60),
            "bid_depth_usd": np.random.lognormal(16, 0.5),
            "ask_depth_usd": np.random.lognormal(16, 0.5),
            "spread_bps": np.random.exponential(5),
            "recent_fill_rate": np.random.uniform(0.7, 0.95)
        })
    
    for inst in instruments:
        sor.update_liquidity_data(
            inst,
            bid_depth=np.random.lognormal(16, 0.5),
            ask_depth=np.random.lognormal(16, 0.5),
            spread_bps=np.random.exponential(5),
            volume=np.random.lognormal(14, 0.5)
        )
    
    # Create routing plans
    print("Creating SOR Routing Plans:\n")
    
    for inst in instruments:
        decision = sor.create_routing_plan(inst, "buy", total_quantity=5e5, urgency=0.5)
        
        print(f"{inst}:")
        print(f"  Total Quantity: ${decision.total_quantity:,.0f}")
        print(f"  Expected Cost: {decision.expected_total_cost_bps:.2f} bps")
        print(f"  Confidence: {decision.confidence_score:.2f}")
        print(f"  Venues: {len(decision.venue_allocations)}")
        
        for alloc in decision.venue_allocations:
            print(f"    - {alloc['venue_id']}: {alloc['allocation_pct']:.1%} @ {alloc['expected_cost_bps']:.1f} bps")
        
        if decision.timing_recommendation.get("should_delay"):
            print(f"  ⚠️  Delay recommended: {decision.timing_recommendation['reason']}")
        
        print()
        
        # Submit and simulate fills
        order_id = sor.submit_routing_plan(decision)
        
        for alloc in decision.venue_allocations[:2]:
            sor.record_fill(
                order_id,
                alloc["venue_id"],
                filled_quantity=alloc["quantity"] * 0.9,
                cost_bps=alloc["expected_cost_bps"] * np.random.uniform(0.8, 1.2)
            )
    
    # Analytics
    analytics = sor.get_sor_analytics()
    print("\nSOR Analytics:")
    print(f"  Total Orders: {analytics['total_orders']}")
    print(f"  Active Orders: {analytics['active_orders']}")
    
    print("\nVenue Performance:")
    for venue, perf in analytics["venue_performance"].items():
        print(f"  {venue}: {perf['avg_cost_bps']:.2f} bps avg ({perf['n_executions']} executions)")
