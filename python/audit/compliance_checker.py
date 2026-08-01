"""
Compliance Checker
Stage 49: Post-trade heuristic compliance checks for wash trades, prohibited venues, and toxic MEV.
"""

import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import zmq

logger = logging.getLogger(__name__)


@dataclass
class ComplianceViolation:
    """Record of a compliance violation."""
    violation_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    description: str
    trade_id: str
    strategy_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)


class ComplianceChecker:
    """
    Performs post-trade heuristic compliance checks.
    Detects wash trades, prohibited venue interactions, and toxic MEV settlements.
    """
    
    def __init__(self,
                 wash_trade_window_seconds: float = 60.0,
                 prohibited_venues: Optional[Set[str]] = None,
                 mev_threshold_usd: float = 1000.0):
        
        self.wash_trade_window = timedelta(seconds=wash_trade_window_seconds)
        self.prohibited_venues = prohibited_venues or set()
        self.mev_threshold = mev_threshold_usd
        
        # Trade history for wash trade detection
        self._trade_history: deque = deque(maxlen=10000)
        
        # Violation history
        self._violations: deque = deque(maxlen=1000)
        
        # Running totals
        self._trades_checked = 0
        self._violations_found = 0
        
        # ZMQ socket for Rust IPC
        self._zmq_context = zmq.Context()
        self._zmq_socket = self._zmq_context.socket(zmq.PUSH)
        self._zmq_socket.connect("tcp://localhost:5568")
    
    async def check_trade(self, trade_data: Dict[str, Any]) -> List[ComplianceViolation]:
        """
        Perform all compliance checks on a trade.
        
        Args:
            trade_data: Dictionary containing trade details
        
        Returns:
            List of violations found (empty if compliant)
        """
        violations = []
        self._trades_checked += 1
        
        # Extract trade details
        trade_id = trade_data.get('trade_id', '')
        strategy_id = trade_data.get('strategy_id', '')
        instrument = trade_data.get('instrument', '')
        side = trade_data.get('side', '')
        quantity = trade_data.get('quantity', 0.0)
        price = trade_data.get('price', 0.0)
        venue = trade_data.get('venue', '')
        timestamp = trade_data.get('timestamp', datetime.utcnow())
        
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        # Check 1: Wash trade detection
        wash_violation = self._check_wash_trade(
            strategy_id, instrument, side, quantity, price, timestamp
        )
        if wash_violation:
            violations.append(wash_violation)
        
        # Check 2: Prohibited venue
        venue_violation = self._check_prohibited_venue(venue, trade_id, strategy_id)
        if venue_violation:
            violations.append(venue_violation)
        
        # Check 3: Toxic MEV detection
        mev_violation = self._check_toxic_mev(trade_data)
        if mev_violation:
            violations.append(mev_violation)
        
        # Record trade for future checks
        self._record_trade(trade_id, strategy_id, instrument, side, quantity, price, timestamp)
        
        # Update counters
        self._violations_found += len(violations)
        
        # Notify Rust of violations
        for violation in violations:
            self._notify_rust(violation)
        
        return violations
    
    def _check_wash_trade(self,
                         strategy_id: str,
                         instrument: str,
                         side: str,
                         quantity: float,
                         price: float,
                         timestamp: datetime) -> Optional[ComplianceViolation]:
        """
        Detect wash trades (buying and selling same instrument within short window).
        """
        # Look for opposite trades in the window
        opposite_side = 'SELL' if side == 'BUY' else 'BUY'
        
        for trade in self._trade_history:
            if trade['strategy_id'] != strategy_id:
                continue
            if trade['instrument'] != instrument:
                continue
            if trade['side'] != opposite_side:
                continue
            
            time_diff = abs((timestamp - trade['timestamp']).total_seconds())
            if time_diff > self.wash_trade_window.total_seconds():
                continue
            
            # Check if quantities match (or are very close)
            if abs(trade['quantity'] - quantity) / max(quantity, 0.001) < 0.1:
                return ComplianceViolation(
                    violation_type="WASH_TRADE",
                    severity="HIGH",
                    description=f"Wash trade detected: {side} {quantity} {instrument} within {time_diff:.1f}s of opposite trade",
                    trade_id="",
                    strategy_id=strategy_id,
                    details={
                        'original_trade_id': trade['trade_id'],
                        'time_diff_seconds': time_diff,
                        'quantity_match': f"{trade['quantity']} vs {quantity}",
                    }
                )
        
        return None
    
    def _check_prohibited_venue(self,
                               venue: str,
                               trade_id: str,
                               strategy_id: str) -> Optional[ComplianceViolation]:
        """Check if trade was executed on a prohibited venue."""
        if venue in self.prohibited_venues:
            return ComplianceViolation(
                violation_type="PROHIBITED_VENUE",
                severity="CRITICAL",
                description=f"Trade executed on prohibited venue: {venue}",
                trade_id=trade_id,
                strategy_id=strategy_id,
                details={'venue': venue}
            )
        
        return None
    
    def _check_toxic_mev(self, trade_data: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """
        Detect potentially toxic MEV (Maximal Extractable Value) patterns.
        Looks for sandwich attacks, front-running indicators, etc.
        """
        # Check for unusual slippage
        expected_price = trade_data.get('expected_price', 0.0)
        actual_price = trade_data.get('price', 0.0)
        
        if expected_price > 0:
            slippage_pct = abs(actual_price - expected_price) / expected_price
            
            if slippage_pct > 0.05:  # More than 5% slippage
                notional = actual_price * trade_data.get('quantity', 0.0)
                
                if notional >= self.mev_threshold:
                    return ComplianceViolation(
                        violation_type="TOXIC_MEV",
                        severity="HIGH",
                        description=f"Potential toxic MEV: {slippage_pct*100:.1f}% slippage on ${notional:,.2f} trade",
                        trade_id=trade_data.get('trade_id', ''),
                        strategy_id=trade_data.get('strategy_id', ''),
                        details={
                            'slippage_pct': slippage_pct,
                            'expected_price': expected_price,
                            'actual_price': actual_price,
                            'notional': notional,
                        }
                    )
        
        # Check for suspicious timing patterns (could indicate front-running)
        # This would require more sophisticated analysis in production
        
        return None
    
    def _record_trade(self,
                     trade_id: str,
                     strategy_id: str,
                     instrument: str,
                     side: str,
                     quantity: float,
                     price: float,
                     timestamp: datetime):
        """Record trade for future compliance checks."""
        self._trade_history.append({
            'trade_id': trade_id,
            'strategy_id': strategy_id,
            'instrument': instrument,
            'side': side,
            'quantity': quantity,
            'price': price,
            'timestamp': timestamp,
        })
    
    def add_prohibited_venue(self, venue: str):
        """Add a venue to the prohibited list."""
        self.prohibited_venues.add(venue)
        logger.info(f"Added prohibited venue: {venue}")
    
    def remove_prohibited_venue(self, venue: str):
        """Remove a venue from the prohibited list."""
        self.prohibited_venues.discard(venue)
        logger.info(f"Removed prohibited venue: {venue}")
    
    def _notify_rust(self, violation: ComplianceViolation):
        """Send violation notification to Rust via ZMQ."""
        try:
            self._zmq_socket.send_json({
                'type': 'COMPLIANCE_VIOLATION',
                'violation_type': violation.violation_type,
                'severity': violation.severity,
                'strategy_id': violation.strategy_id,
                'description': violation.description,
                'timestamp': violation.timestamp.isoformat(),
            }, flags=zmq.NOBLOCK)
        except Exception as e:
            logger.error(f"Failed to send violation to Rust: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get compliance checker status."""
        return {
            'trades_checked': self._trades_checked,
            'violations_found': self._violations_found,
            'prohibited_venues': list(self.prohibited_venues),
            'recent_violations': [
                {
                    'type': v.violation_type,
                    'severity': v.severity,
                    'timestamp': v.timestamp.isoformat(),
                }
                for v in list(self._violations)[-10:]
            ],
        }
    
    def shutdown(self):
        """Cleanup resources."""
        self._zmq_socket.close()
        self._zmq_context.term()
        logger.info("ComplianceChecker shut down")


# Global instance
_checker: Optional[ComplianceChecker] = None


def get_checker() -> ComplianceChecker:
    """Get or create the global ComplianceChecker instance."""
    global _checker
    if _checker is None:
        _checker = ComplianceChecker()
    return _checker


def create_checker(wash_trade_window_seconds: float = 60.0,
                  prohibited_venues: Optional[Set[str]] = None,
                  mev_threshold_usd: float = 1000.0) -> ComplianceChecker:
    """Create a new ComplianceChecker with custom configuration."""
    global _checker
    _checker = ComplianceChecker(
        wash_trade_window_seconds=wash_trade_window_seconds,
        prohibited_venues=prohibited_venues,
        mev_threshold_usd=mev_threshold_usd,
    )
    return _checker
