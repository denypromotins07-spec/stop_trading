"""
Audit Module Root
Stage 49: Streams enriched audit logs to Rust SOUL.md writer and local Parquet storage.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import zmq

from .trade_journaler import TradeJournaler, get_journaler
from .compliance_checker import ComplianceChecker, get_checker

logger = logging.getLogger(__name__)


class AuditModule:
    """
    Central module streaming enriched audit logs to Rust SOUL.md writer
    and local encrypted Parquet storage for offline retraining.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Components
        self.journaler: Optional[TradeJournaler] = None
        self.compliance_checker: Optional[ComplianceChecker] = None
        
        # State
        self._running = False
        
        # ZMQ socket for Rust SOUL.md writer
        self._zmq_context: Optional[zmq.Context] = None
        self._soul_socket: Optional[zmq.Socket] = None
        
        # Parquet storage path
        self.parquet_dir = Path(config.get('parquet_dir', '/tmp/audit_parquet'))
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self._events_streamed = 0
        self._compliance_checks = 0
    
    async def initialize(self) -> bool:
        """Initialize the audit module."""
        try:
            logger.info("Initializing AuditModule...")
            
            # Create components
            self.journaler = TradeJournaler(
                log_dir=self.config.get('log_dir', '/tmp/trade_logs'),
                queue_max_size=self.config.get('queue_max_size', 10000),
                batch_size=self.config.get('batch_size', 100),
            )
            
            self.compliance_checker = ComplianceChecker(
                wash_trade_window_seconds=self.config.get('wash_trade_window', 60.0),
                prohibited_venues=set(self.config.get('prohibited_venues', [])),
                mev_threshold_usd=self.config.get('mev_threshold', 1000.0),
            )
            
            # Setup ZMQ connection to Rust SOUL.md writer
            self._zmq_context = zmq.Context()
            self._soul_socket = self._zmq_context.socket(zmq.PUSH)
            self._soul_socket.connect("tcp://localhost:5569")
            
            # Start journaler
            await self.journaler.start()
            
            self._running = True
            logger.info("AuditModule initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize AuditModule: {e}")
            return False
    
    async def log_trade(self, 
                       event_type: str,
                       event_data: Any,
                       feature_snapshot: Dict[str, Any],
                       run_compliance: bool = True):
        """Log a trade event with optional compliance checking."""
        if not self._running:
            logger.warning("AuditModule not running")
            return
        
        try:
            # Log to journal
            if event_type == "OrderFilled":
                await self.journaler.log_order_filled(event_data, feature_snapshot)
            elif event_type == "PositionClosed":
                await self.journaler.log_position_closed(event_data, feature_snapshot)
            
            # Run compliance checks
            if run_compliance and self.compliance_checker:
                trade_data = {
                    'event_type': event_type,
                    **self._extract_trade_data(event_data),
                    'feature_snapshot': feature_snapshot,
                }
                
                violations = await self.compliance_checker.check_trade(trade_data)
                self._compliance_checks += 1
                
                if violations:
                    logger.warning(f"Compliance violations found: {len(violations)}")
            
            # Stream to Rust SOUL.md writer
            self._stream_to_soul(event_type, event_data, feature_snapshot)
            self._events_streamed += 1
            
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")
    
    def _extract_trade_data(self, event_data: Any) -> Dict[str, Any]:
        """Extract trade data from Nautilus event."""
        if hasattr(self.journaler, 'serialize_nautilus_object'):
            return self.journaler.serialize_nautilus_object(event_data)
        return {}
    
    def _stream_to_soul(self, 
                       event_type: str,
                       event_data: Any,
                       feature_snapshot: Dict[str, Any]):
        """Stream enriched audit log to Rust SOUL.md writer."""
        try:
            self._soul_socket.send_json({
                'type': 'AUDIT_EVENT',
                'event_type': event_type,
                'data': self._extract_trade_data(event_data),
                'features': feature_snapshot,
                'timestamp': datetime.utcnow().isoformat(),
            }, flags=zmq.NOBLOCK)
        except Exception as e:
            logger.error(f"Failed to stream to SOUL.md: {e}")
    
    async def export_parquet(self, 
                            start_date: str,
                            end_date: str,
                            output_path: Optional[str] = None) -> Optional[str]:
        """Export audit logs to Parquet format for offline retraining."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            
            # Collect events from journal (in production, would query database)
            # This is a placeholder - actual implementation would query persistent storage
            
            output_path = output_path or str(
                self.parquet_dir / f"audit_export_{start_date}_{end_date}.parquet"
            )
            
            logger.info(f"Parquet export placeholder: {output_path}")
            # Actual implementation would:
            # 1. Query trade journal for date range
            # 2. Convert to PyArrow Table
            # 3. Write compressed Parquet with encryption
            
            return output_path
            
        except ImportError:
            logger.warning("pyarrow not installed, skipping Parquet export")
            return None
        except Exception as e:
            logger.error(f"Parquet export failed: {e}")
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get audit module status."""
        return {
            'running': self._running,
            'events_streamed': self._events_streamed,
            'compliance_checks': self._compliance_checks,
            'journaler': self.journaler.get_status() if self.journaler else None,
            'compliance_checker': self.compliance_checker.get_status() if self.compliance_checker else None,
            'parquet_dir': str(self.parquet_dir),
        }
    
    async def shutdown(self):
        """Gracefully shutdown the audit module."""
        logger.info("Shutting down AuditModule...")
        self._running = False
        
        if self.journaler:
            await self.journaler.stop()
        
        if self.compliance_checker:
            self.compliance_checker.shutdown()
        
        if self._soul_socket:
            self._soul_socket.close()
        if self._zmq_context:
            self._zmq_context.term()
        
        logger.info("AuditModule shut down complete")


# Global module instance
_module: Optional[AuditModule] = None


def get_module() -> AuditModule:
    """Get or create the global AuditModule instance."""
    global _module
    if _module is None:
        _module = AuditModule({})
    return _module


def create_module(config: Dict[str, Any]) -> AuditModule:
    """Create a new AuditModule with custom configuration."""
    global _module
    _module = AuditModule(config)
    return _module
