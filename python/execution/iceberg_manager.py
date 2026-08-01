"""
Chapter 2: Advanced Execution Orchestration & Smart Scheduling
File: python/execution/iceberg_manager.py

Python-side iceberg order manager that syncs with Rust L3 queue tracker.
Dynamically adjusts visible clip size and refresh latency to hide true footprint
from predatory HFT sniffers.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from datetime import datetime, timedelta
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class IcebergState(Enum):
    ACTIVE = "active"
    PARTIALLY_FILLED = "partially_filled"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class IcebergConfig:
    """Configuration for iceberg order management."""
    # Order parameters
    total_quantity: float = 10000.0
    min_clip_size: float = 50.0
    max_clip_size: float = 500.0
    initial_clip_size: float = 100.0
    
    # Anti-sniffing parameters
    randomize_clip_size: bool = True
    clip_size_variance_pct: float = 0.2  # 20% variance
    randomize_refresh_delay: bool = True
    refresh_delay_ms_range: Tuple[int, int] = (50, 500)
    
    # Dynamic adjustment parameters
    fill_rate_threshold_fast: float = 0.8  # Reduce clip if >80% filled quickly
    fill_rate_threshold_slow: float = 0.2  # Increase clip if <20% filled slowly
    adaptation_factor: float = 0.1
    
    # Safety limits
    max_participation_rate: float = 0.15  # Max 15% of market volume
    max_duration_minutes: int = 120
    price_tolerance_bps: float = 50.0  # Cancel if price moves >50bps


@dataclass
class IcebergClip:
    """Represents a single visible clip of the iceberg."""
    clip_id: int
    side: OrderSide
    quantity: float
    remaining_quantity: float
    limit_price: float
    submit_time: datetime
    fill_time: Optional[datetime] = None
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    status: str = "pending"


@dataclass
class L3QueueState:
    """L3 order book queue state from Rust tracker."""
    best_bid: float = 0.0
    best_ask: float = 0.0
    bid_queue_size: float = 0.0
    ask_queue_size: float = 0.0
    recent_trade_volume: float = 0.0
    trade_flow_rate: float = 0.0  # Trades per second
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def get_mid_price(self) -> float:
        if self.best_bid > 0 and self.best_ask > 0:
            return (self.best_bid + self.best_ask) / 2
        return self.best_bid or self.best_ask
    
    def get_spread_bps(self) -> float:
        if self.get_mid_price() > 0:
            return (self.best_ask - self.best_bid) / self.get_mid_price() * 10000
        return 0.0


class IcebergOrderManager:
    """
    Manages iceberg orders with dynamic clip sizing and anti-sniffing measures.
    Syncs with Rust L3 queue tracker for optimal placement.
    """
    
    def __init__(self, config: Optional[IcebergConfig] = None):
        self.config = config or IcebergConfig()
        self.state = IcebergState.ACTIVE
        
        # Order tracking
        self.total_quantity = self.config.total_quantity
        self.remaining_quantity = self.total_quantity
        self.filled_quantity = 0.0
        self.avg_fill_price = 0.0
        
        # Clip management
        self.clips: List[IcebergClip] = []
        self.current_clip_idx = 0
        self.active_clip: Optional[IcebergClip] = None
        
        # Dynamic sizing state
        self.current_clip_size = self.config.initial_clip_size
        self.fill_rate_history: List[float] = []
        
        # L3 sync state
        self.l3_state: Optional[L3QueueState] = None
        self.last_l3_update: Optional[datetime] = None
        
        # Timing
        self.start_time: Optional[datetime] = None
        self.last_clip_time: Optional[datetime] = None
        
        # Callbacks for Rust integration
        self.submit_callback: Optional[callable] = None
        self.cancel_callback: Optional[callable] = None
        self.l3_sync_callback: Optional[callable] = None
    
    def initialize(
        self,
        total_quantity: float,
        side: OrderSide,
        limit_price: float,
        start_time: Optional[datetime] = None
    ):
        """Initialize iceberg order."""
        self.config.total_quantity = total_quantity
        self.total_quantity = total_quantity
        self.remaining_quantity = total_quantity
        self.side = side
        self.limit_price = limit_price
        self.start_time = start_time or datetime.utcnow()
        
        self.clips.clear()
        self.current_clip_idx = 0
        self.filled_quantity = 0.0
        self.avg_fill_price = 0.0
        self.fill_rate_history.clear()
        
        self.state = IcebergState.ACTIVE
        
        logger.info(
            f"Iceberg initialized: {total_quantity} {side.value} @ {limit_price}"
        )
    
    def sync_l3_state(self, l3_state: L3QueueState):
        """Sync with Rust L3 queue tracker."""
        self.l3_state = l3_state
        self.last_l3_update = datetime.utcnow()
    
    async def fetch_l3_state(self) -> L3QueueState:
        """Fetch latest L3 state from Rust tracker."""
        if self.l3_sync_callback:
            self.l3_state = await self.l3_sync_callback()
            self.last_l3_update = datetime.utcnow()
        return self.l3_state
    
    def calculate_adaptive_clip_size(self) -> float:
        """
        Calculate adaptive clip size based on fill rate and market conditions.
        Implements anti-sniffing randomization.
        """
        base_size = self.current_clip_size
        
        # Adjust based on fill rate history
        if len(self.fill_rate_history) >= 3:
            avg_fill_rate = np.mean(self.fill_rate_history[-3:])
            
            if avg_fill_rate > self.config.fill_rate_threshold_fast:
                # Filling too fast - reduce clip size to hide footprint
                base_size *= (1 - self.config.adaptation_factor)
            elif avg_fill_rate < self.config.fill_rate_threshold_slow:
                # Filling too slow - increase clip size for efficiency
                base_size *= (1 + self.config.adaptation_factor)
        
        # Apply randomization for anti-sniffing
        if self.config.randomize_clip_size:
            variance = base_size * self.config.clip_size_variance_pct
            random_adjustment = random.uniform(-variance, variance)
            base_size += random_adjustment
        
        # Clamp to bounds
        clip_size = np.clip(
            base_size,
            self.config.min_clip_size,
            min(self.config.max_clip_size, self.remaining_quantity)
        )
        
        # Update current clip size for next iteration
        self.current_clip_size = clip_size
        
        return clip_size
    
    def calculate_refresh_delay(self) -> float:
        """Calculate randomized refresh delay in milliseconds."""
        if self.config.randomize_refresh_delay:
            return random.uniform(
                self.config.refresh_delay_ms_range[0],
                self.config.refresh_delay_ms_range[1]
            )
        return np.mean(self.config.refresh_delay_ms_range)
    
    def create_next_clip(self) -> Optional[IcebergClip]:
        """Create the next iceberg clip."""
        if self.remaining_quantity <= 0:
            self.state = IcebergState.COMPLETED
            return None
        
        # Calculate adaptive clip size
        clip_size = self.calculate_adaptive_clip_size()
        
        # Ensure we don't exceed remaining quantity
        clip_size = min(clip_size, self.remaining_quantity)
        
        # Create clip
        clip = IcebergClip(
            clip_id=len(self.clips),
            side=self.side,
            quantity=clip_size,
            remaining_quantity=clip_size,
            limit_price=self.limit_price,
            submit_time=datetime.utcnow()
        )
        
        self.clips.append(clip)
        self.active_clip = clip
        self.last_clip_time = datetime.utcnow()
        
        logger.debug(
            f"Created clip {clip.clip_id}: {clip_size} @ {self.limit_price}"
        )
        
        return clip
    
    async def submit_clip(self, clip: IcebergClip) -> bool:
        """Submit clip to exchange via Rust execution engine."""
        if not self.submit_callback:
            logger.warning("No submit callback registered - simulating")
            return True
        
        try:
            result = await self.submit_callback(
                side=clip.side.value,
                quantity=clip.quantity,
                limit_price=clip.limit_price,
                is_iceberg=True,
                display_qty=clip.quantity  # Visible portion
            )
            return result.get("success", False)
        except Exception as e:
            logger.error(f"Failed to submit clip: {e}")
            return False
    
    async def cancel_active_clip(self) -> bool:
        """Cancel the active clip."""
        if not self.active_clip:
            return True
        
        if not self.cancel_callback:
            logger.warning("No cancel callback registered - simulating")
            self.active_clip.status = "cancelled"
            return True
        
        try:
            result = await self.cancel_callback(
                clip_id=self.active_clip.clip_id
            )
            if result.get("success", False):
                self.active_clip.status = "cancelled"
            return result.get("success", False)
        except Exception as e:
            logger.error(f"Failed to cancel clip: {e}")
            return False
    
    def update_fill(
        self, 
        clip_id: int, 
        filled_quantity: float, 
        fill_price: float
    ):
        """Update fill status for a clip."""
        for clip in self.clips:
            if clip.clip_id == clip_id:
                old_remaining = clip.remaining_quantity
                clip.filled_quantity += filled_quantity
                clip.remaining_quantity -= filled_quantity
                
                # Update average fill price
                total_value = (
                    self.filled_quantity * self.avg_fill_price +
                    filled_quantity * fill_price
                )
                self.filled_quantity += filled_quantity
                self.avg_fill_price = total_value / self.filled_quantity
                
                # Update remaining quantity
                self.remaining_quantity -= filled_quantity
                
                # Calculate fill rate
                if old_remaining > 0:
                    fill_rate = filled_quantity / old_remaining
                    self.fill_rate_history.append(fill_rate)
                
                # Update clip status
                if clip.remaining_quantity <= 0:
                    clip.status = "filled"
                    clip.fill_time = datetime.utcnow()
                else:
                    clip.status = "partially_filled"
                
                logger.debug(
                    f"Clip {clip_id} fill: {filled_quantity} @ {fill_price}, "
                    f"remaining: {clip.remaining_quantity}"
                )
                
                # Check if order complete
                if self.remaining_quantity <= 0:
                    self.state = IcebergState.COMPLETED
                
                break
    
    async def run(
        self,
        side: OrderSide,
        limit_price: float,
        check_interval_ms: int = 100
    ):
        """
        Run the iceberg order manager.
        
        Args:
            side: Buy or sell
            limit_price: Limit price for all clips
            check_interval_ms: How often to check fill status
        """
        self.initialize(self.total_quantity, side, limit_price)
        
        logger.info("Iceberg order manager started")
        
        while self.state not in [IcebergState.COMPLETED, IcebergState.CANCELLED]:
            # Check timeout
            elapsed = (datetime.utcnow() - self.start_time).total_seconds() / 60
            if elapsed > self.config.max_duration_minutes:
                logger.warning("Iceberg timeout exceeded")
                self.state = IcebergState.CANCELLED
                break
            
            # If no active clip, create and submit one
            if not self.active_clip or self.active_clip.remaining_quantity <= 0:
                # Wait for randomized refresh delay
                if self.last_clip_time:
                    delay_ms = self.calculate_refresh_delay()
                    await asyncio.sleep(delay_ms / 1000)
                
                # Fetch latest L3 state
                await self.fetch_l3_state()
                
                # Create next clip
                clip = self.create_next_clip()
                if not clip:
                    break
                
                # Submit clip
                success = await self.submit_clip(clip)
                if not success:
                    logger.error("Failed to submit clip, retrying...")
                    await asyncio.sleep(check_interval_ms / 1000)
                    continue
            
            # Wait for next check
            await asyncio.sleep(check_interval_ms / 1000)
        
        logger.info(
            f"Iceberg completed: {self.filled_quantity}/{self.total_quantity} "
            f"@ avg price {self.avg_fill_price:.4f}"
        )
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current iceberg progress."""
        completion_pct = (
            self.filled_quantity / self.total_quantity * 100
            if self.total_quantity > 0 else 0.0
        )
        
        return {
            "state": self.state.value,
            "completion_pct": completion_pct,
            "total_quantity": self.total_quantity,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "avg_fill_price": self.avg_fill_price,
            "clips_submitted": len(self.clips),
            "current_clip_size": self.current_clip_size,
            "avg_fill_rate": np.mean(self.fill_rate_history) if self.fill_rate_history else 0.0,
            "l3_last_update": self.last_l3_update.isoformat() if self.last_l3_update else None
        }
    
    def pause(self):
        """Pause iceberg execution."""
        self.state = IcebergState.PAUSED
        logger.info("Iceberg paused")
    
    def resume(self):
        """Resume paused iceberg execution."""
        if self.state == IcebergState.PAUSED:
            self.state = IcebergState.ACTIVE
            logger.info("Iceberg resumed")
    
    def cancel(self):
        """Cancel iceberg order."""
        self.state = IcebergState.CANCELLED
        logger.info("Iceberg cancelled")


# Export for module use
__all__ = [
    "OrderSide",
    "IcebergState",
    "IcebergConfig",
    "IcebergClip",
    "L3QueueState",
    "IcebergOrderManager"
]
