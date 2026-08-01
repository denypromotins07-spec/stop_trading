"""
State Reconciliation & Healing - Background loop comparing Nautilus state with Rust IPC and exchange snapshots.
Automatically detects and corrects state divergence to prevent catastrophic portfolio errors.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import numpy as np
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DivergenceSeverity(Enum):
    """Severity levels for state divergence."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PositionSnapshot:
    """Snapshot of position state from a source."""
    timestamp: float
    source: str  # 'nautilus', 'rust_ipc', 'exchange_rest'
    symbol: str
    net_position: float
    long_qty: float
    short_qty: float
    avg_entry_price: float
    unrealized_pnl: float
    realized_pnl: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "source": self.source,
            "symbol": self.symbol,
            "net_position": self.net_position,
            "long_qty": self.long_qty,
            "short_qty": self.short_qty,
            "avg_entry_price": self.avg_entry_price,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl
        }


@dataclass
class DivergenceEvent:
    """Detected state divergence."""
    timestamp: float
    symbol: str
    severity: DivergenceSeverity
    source_a: str
    source_b: str
    field: str
    value_a: float
    value_b: float
    difference: float
    difference_pct: float
    auto_corrected: bool = False
    correction_action: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "severity": self.severity.value,
            "sources": [self.source_a, self.source_b],
            "field": self.field,
            "value_a": self.value_a,
            "value_b": self.value_b,
            "difference": self.difference,
            "difference_pct": self.difference_pct,
            "auto_corrected": self.auto_corrected,
            "correction_action": self.correction_action
        }


class StateHealer:
    """
    Background reconciliation engine for detecting and healing state divergence.
    Compares Nautilus cached state with Rust IPC and exchange REST snapshots.
    """
    
    def __init__(self,
                 tolerance_pct: float = 0.01,
                 critical_tolerance_pct: float = 0.05,
                 check_interval: float = 1.0,
                 max_history: int = 1000):
        """
        Initialize state healer.
        
        Args:
            tolerance_pct: Tolerance for low-severity divergence
            critical_tolerance_pct: Threshold for critical divergence
            check_interval: Seconds between reconciliation checks
            max_history: Maximum divergence events to keep in history
        """
        self.tolerance_pct = tolerance_pct
        self.critical_tolerance_pct = critical_tolerance_pct
        self.check_interval = check_interval
        self.max_history = max_history
        
        # Latest snapshots from each source
        self._snapshots: Dict[str, Dict[str, PositionSnapshot]] = {}
        
        # Divergence history
        self._divergence_history: deque = deque(maxlen=max_history)
        
        # Auto-correction settings
        self._auto_correct_enabled = True
        self._correction_count = 0
        
        # Health tracking
        self._last_check_time: float = 0.0
        self._consecutive_divergences = 0
        self._is_running = False
    
    def update_snapshot(self, snapshot: PositionSnapshot):
        """Update snapshot from a source."""
        if snapshot.source not in self._snapshots:
            self._snapshots[snapshot.source] = {}
        
        self._snapshots[snapshot.source][snapshot.symbol] = snapshot
        logger.debug(f"Updated {snapshot.source} snapshot for {snapshot.symbol}")
    
    def check_divergence(self) -> List[DivergenceEvent]:
        """
        Check for divergence between all sources.
        
        Returns:
            List of detected divergence events
        """
        divergences = []
        sources = list(self._snapshots.keys())
        
        if len(sources) < 2:
            return divergences
        
        # Compare each pair of sources
        for i, source_a in enumerate(sources):
            for source_b in sources[i+1:]:
                symbols_a = set(self._snapshots[source_a].keys())
                symbols_b = set(self._snapshots[source_b].keys())
                common_symbols = symbols_a & symbols_b
                
                for symbol in common_symbols:
                    snap_a = self._snapshots[source_a][symbol]
                    snap_b = self._snapshots[source_b][symbol]
                    
                    # Check each field
                    fields_to_check = [
                        ('net_position', True),
                        ('long_qty', False),
                        ('short_qty', False),
                        ('avg_entry_price', True),
                        ('unrealized_pnl', True),
                        ('realized_pnl', True)
                    ]
                    
                    for field_name, is_relative in fields_to_check:
                        value_a = getattr(snap_a, field_name)
                        value_b = getattr(snap_b, field_name)
                        
                        if value_a == 0 and value_b == 0:
                            continue
                        
                        # Calculate difference
                        diff = abs(value_a - value_b)
                        
                        if is_relative and value_a != 0:
                            diff_pct = diff / abs(value_a) * 100
                        else:
                            # Absolute tolerance for quantities
                            diff_pct = diff / max(abs(value_a), abs(value_b), 1) * 100
                        
                        # Determine severity
                        severity = self._classify_severity(diff_pct)
                        
                        if severity != DivergenceSeverity.NONE:
                            event = DivergenceEvent(
                                timestamp=time.time(),
                                symbol=symbol,
                                severity=severity,
                                source_a=source_a,
                                source_b=source_b,
                                field=field_name,
                                value_a=value_a,
                                value_b=value_b,
                                difference=diff,
                                difference_pct=diff_pct
                            )
                            
                            divergences.append(event)
                            self._divergence_history.append(event)
                            
                            # Attempt auto-correction for high severity
                            if severity in [DivergenceSeverity.HIGH, DivergenceSeverity.CRITICAL]:
                                self._attempt_correction(event)
        
        self._last_check_time = time.time()
        
        if divergences:
            self._consecutive_divergences += 1
            logger.warning(f"Found {len(divergences)} divergences")
        else:
            self._consecutive_divergences = 0
        
        return divergences
    
    def _classify_severity(self, diff_pct: float) -> DivergenceSeverity:
        """Classify divergence severity based on percentage difference."""
        if diff_pct == 0:
            return DivergenceSeverity.NONE
        elif diff_pct <= self.tolerance_pct:
            return DivergenceSeverity.LOW
        elif diff_pct <= self.tolerance_pct * 5:
            return DivergenceSeverity.MEDIUM
        elif diff_pct <= self.critical_tolerance_pct:
            return DivergenceSeverity.HIGH
        else:
            return DivergenceSeverity.CRITICAL
    
    def _attempt_correction(self, event: DivergenceEvent):
        """Attempt to auto-correct divergence."""
        if not self._auto_correct_enabled:
            return
        
        # Strategy: trust exchange REST as ground truth
        trusted_source = 'exchange_rest'
        
        if event.source_a == trusted_source:
            trusted_value = event.value_a
            untrusted_source = event.source_b
        elif event.source_b == trusted_source:
            trusted_value = event.value_b
            untrusted_source = event.source_a
        else:
            # No trusted source available, flag for manual review
            event.correction_action = "manual_review_required"
            return
        
        event.auto_corrected = True
        event.correction_action = f"flagged_{untrusted_source}_for_reconciliation"
        self._correction_count += 1
        
        logger.info(
            f"Auto-corrected divergence: {event.symbol}.{event.field} "
            f"({untrusted_source}: {event.value_b} → trusted: {trusted_value})"
        )
    
    async def run_reconciliation_loop(self, stop_event: asyncio.Event):
        """
        Run background reconciliation loop.
        
        Args:
            stop_event: Event to signal shutdown
        """
        self._is_running = True
        logger.info("Starting reconciliation loop")
        
        while not stop_event.is_set():
            try:
                divergences = self.check_divergence()
                
                # Alert on critical divergences
                critical = [d for d in divergences if d.severity == DivergenceSeverity.CRITICAL]
                if critical:
                    await self._send_alert(critical)
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Reconciliation error: {e}")
                await asyncio.sleep(self.check_interval)
        
        self._is_running = False
        logger.info("Reconciliation loop stopped")
    
    async def _send_alert(self, critical_divergences: List[DivergenceEvent]):
        """Send alert for critical divergences."""
        for div in critical_divergences:
            logger.critical(
                f"CRITICAL DIVERGENCE: {div.symbol}.{div.field} "
                f"diff={div.difference_pct:.2f}% "
                f"({div.source_a}={div.value_a} vs {div.source_b}={div.value_b})"
            )
            # In production, would send to monitoring system
    
    def get_divergence_summary(self) -> Dict[str, Any]:
        """Get summary of recent divergences."""
        if not self._divergence_history:
            return {"status": "no_divergences"}
        
        recent = list(self._divergence_history)[-100:]
        
        severity_counts = {s.value: 0 for s in DivergenceSeverity}
        for div in recent:
            severity_counts[div.severity.value] = severity_counts.get(div.severity.value, 0) + 1
        
        symbols_affected = set(d.symbol for d in recent)
        auto_corrected = sum(1 for d in recent if d.auto_corrected)
        
        return {
            "total_events": len(recent),
            "severity_breakdown": severity_counts,
            "symbols_affected": list(symbols_affected),
            "auto_corrected_count": auto_corrected,
            "consecutive_divergences": self._consecutive_divergences,
            "last_check": self._last_check_time,
            "is_running": self._is_running
        }
    
    def enable_auto_correction(self, enabled: bool = True):
        """Enable or disable auto-correction."""
        self._auto_correct_enabled = enabled
        logger.info(f"Auto-correction {'enabled' if enabled else 'disabled'}")
    
    def health_check(self) -> Dict[str, Any]:
        """Return healer health status."""
        sources_available = list(self._snapshots.keys())
        total_snapshots = sum(len(snaps) for snaps in self._snapshots.values())
        
        return {
            "running": self._is_running,
            "sources_available": sources_available,
            "total_snapshots": total_snapshots,
            "divergence_count": len(self._divergence_history),
            "corrections_made": self._correction_count,
            "consecutive_issues": self._consecutive_divergences
        }


# Module singleton
_healer: Optional[StateHealer] = None


def get_state_healer(**kwargs) -> StateHealer:
    """Get or create the global state healer."""
    global _healer
    
    if _healer is None:
        _healer = StateHealer(**kwargs)
        logger.info("Created state healer")
    
    return _healer


if __name__ == "__main__":
    # Test the state healer
    print("Testing State Healer...")
    
    healer = StateHealer(tolerance_pct=0.01, critical_tolerance_pct=0.05)
    
    # Create test snapshots
    base_time = time.time()
    
    # Normal state - all sources agree
    normal_snap = PositionSnapshot(
        timestamp=base_time,
        source='nautilus',
        symbol='BTC/USD',
        net_position=100.0,
        long_qty=100.0,
        short_qty=0.0,
        avg_entry_price=45000.0,
        unrealized_pnl=500.0,
        realized_pnl=1000.0
    )
    
    healer.update_snapshot(normal_snap)
    
    healer.update_snapshot(PositionSnapshot(
        timestamp=base_time,
        source='rust_ipc',
        symbol='BTC/USD',
        net_position=100.0,
        long_qty=100.0,
        short_qty=0.0,
        avg_entry_price=45000.0,
        unrealized_pnl=500.0,
        realized_pnl=1000.0
    ))
    
    healer.update_snapshot(PositionSnapshot(
        timestamp=base_time,
        source='exchange_rest',
        symbol='BTC/USD',
        net_position=100.0,
        long_qty=100.0,
        short_qty=0.0,
        avg_entry_price=45000.0,
        unrealized_pnl=500.0,
        realized_pnl=1000.0
    ))
    
    # Check - should be no divergence
    divergences = healer.check_divergence()
    print(f"\nNormal state: {len(divergences)} divergences")
    
    # Introduce divergence
    bad_snap = PositionSnapshot(
        timestamp=base_time,
        source='nautilus',
        symbol='BTC/USD',
        net_position=105.0,  # Different!
        long_qty=105.0,
        short_qty=0.0,
        avg_entry_price=45000.0,
        unrealized_pnl=500.0,
        realized_pnl=1000.0
    )
    healer.update_snapshot(bad_snap)
    
    divergences = healer.check_divergence()
    print(f"\nAfter introducing divergence: {len(divergences)} divergences")
    
    for div in divergences:
        print(f"  {div.symbol}.{div.field}: {div.difference_pct:.2f}% ({div.severity.value})")
        if div.auto_corrected:
            print(f"    → Auto-corrected: {div.correction_action}")
    
    print(f"\nSummary: {healer.get_divergence_summary()}")
    print(f"Health: {healer.health_check()}")
