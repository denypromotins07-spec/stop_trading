# Rust Bridge Adapter: LiveDataClient reading shared memory ring buffer
# Converts zero-copy numpy arrays into Nautilus CustomData events

from __future__ import annotations
import asyncio
import mmap
import logging
from typing import Optional

import numpy as np
from nautilus_trader.adapters.base import LiveDataClient
from nautilus_trader.core.data import Data
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.msgbus.bus import MessageBus
from nautilus_trader.common.component import Clock

from python.nautilus_data.custom_types import OrderFlowData, SMCBlockData, RegimeStateData
from python.ipc_bridge.schema_parser import (
    RustOrderFlowHeader,
    RustSMCBlockHeader,
    RustRegimeStateHeader,
    SHM_CONFIG,
)

log = logging.getLogger(__name__)


class RustBridgeAdapter(LiveDataClient):
    """
    LiveDataClient adapter that reads Rust shared memory ring buffer.
    Publishes zero-copy converted CustomData events to Nautilus MessageBus.
    """

    def __init__(
        self,
        msgbus: MessageBus,
        clock: Clock,
        shm_path: str = "/tmp/rust_hft_shm",
        instrument_id: Optional[InstrumentId] = None,
    ) -> None:
        super().__init__(msgbus=msgbus, clock=clock)
        self.shm_path = shm_path
        self.instrument_id = instrument_id or InstrumentId.from_str("BTCUSDT.BINANCE")
        self._mm: Optional[mmap.mmap] = None
        self._header_size = 64  # Fixed header size in bytes
        self._running = False
        self._last_offset = 0
        
        # Pre-allocated buffers for zero-copy parsing
        self._order_flow_buffer = np.empty(100, dtype=np.float64)
        self._smc_buffer = np.empty(50, dtype=np.float64)
        self._regime_buffer = np.empty(10, dtype=np.float64)

    async def _connect(self) -> None:
        """Initialize shared memory mapping."""
        try:
            self._mm = mmap.mmap(-1, SHM_CONFIG["total_size"], tagname="rust_hft_shm")
            log.info(f"Connected to Rust shared memory at {self.shm_path}")
        except Exception as e:
            log.warning(f"Shared memory not available yet: {e}. Retrying...")
            await asyncio.sleep(0.1)
            await self._connect()

    async def _disconnect(self) -> None:
        """Cleanup shared memory mapping."""
        if self._mm:
            self._mm.close()
            self._mm = None
        self._running = False
        log.info("Disconnected from Rust shared memory")

    def _parse_order_flow(self, offset: int) -> Optional[OrderFlowData]:
        """Parse order flow data from shared memory using zero-copy view."""
        if not self._mm:
            return None
            
        try:
            # Create memoryview for zero-copy access
            mv = memoryview(self._mm)[offset:offset + self._header_size]
            header = np.frombuffer(mv, dtype=np.float64, count=8)
            
            if header[0] < 1e-9:  # Empty slot marker
                return None
                
            return OrderFlowData(
                instrument_id=self.instrument_id,
                ts_event=int(header[1]),
                ts_init=int(header[2]),
                aggressor_side="BUY" if header[3] > 0 else "SELL",
                volume=float(header[4]),
                price=float(header[5]),
                micro_price_deviation=float(header[6]),
                trade_count=int(header[7]),
            )
        except Exception as e:
            log.error(f"Error parsing order flow: {e}")
            return None

    def _parse_smc_block(self, offset: int) -> Optional[SMCBlockData]:
        """Parse SMC block data from shared memory."""
        if not self._mm:
            return None
            
        try:
            mv = memoryview(self._mm)[offset:offset + self._header_size]
            header = np.frombuffer(mv, dtype=np.float64, count=8)
            
            if header[0] < 1e-9:
                return None
                
            block_types = ["BULLISH_OB", "BEARISH_OB", "FVG", "LIQUIDITY"]
            block_idx = int(header[3]) % len(block_types)
            
            return SMCBlockData(
                instrument_id=self.instrument_id,
                ts_event=int(header[1]),
                ts_init=int(header[2]),
                block_type=block_types[block_idx],
                start_price=float(header[4]),
                end_price=float(header[5]),
                strength=float(header[6]),
                touched=bool(int(header[7])),
            )
        except Exception as e:
            log.error(f"Error parsing SMC block: {e}")
            return None

    def _parse_regime_state(self, offset: int) -> Optional[RegimeStateData]:
        """Parse regime state data from shared memory."""
        if not self._mm:
            return None
            
        try:
            mv = memoryview(self._mm)[offset:offset + self._header_size]
            header = np.frombuffer(mv, dtype=np.float64, count=8)
            
            if header[0] < 1e-9:
                return None
                
            regime_types = ["TRENDING_UP", "TRENDING_DOWN", "MEAN_REVERTING", "HIGH_VOL", "LOW_VOL"]
            regime_idx = int(header[3]) % len(regime_types)
            
            return RegimeStateData(
                instrument_id=self.instrument_id,
                ts_event=int(header[1]),
                ts_init=int(header[2]),
                regime_type=regime_types[regime_idx],
                confidence=float(header[4]),
                volatility=float(header[5]),
                trend_strength=float(header[6]),
                mean_reversion_score=float(header[7]),
            )
        except Exception as e:
            log.error(f"Error parsing regime state: {e}")
            return None

    async def _poll_shm_loop(self) -> None:
        """Main polling loop for shared memory updates."""
        self._running = True
        poll_interval = 0.0001  # 100 microseconds
        
        while self._running:
            if not self._mm:
                await asyncio.sleep(poll_interval)
                continue
                
            try:
                # Read current write offset from header (first 8 bytes)
                mv_header = memoryview(self._mm)[0:8]
                current_offset = int(np.frombuffer(mv_header, dtype=np.int64, count=1)[0])
                
                if current_offset != self._last_offset:
                    # New data available, parse and publish
                    data_offset = self._header_size + (current_offset * SHM_CONFIG["slot_size"])
                    
                    # Parse different data types based on type marker
                    type_marker = int(np.frombuffer(memoryview(self._mm)[data_offset:data_offset+8], 
                                                    dtype=np.int64, count=1)[0])
                    
                    if type_marker == 1:  # Order Flow
                        data = self._parse_order_flow(data_offset)
                    elif type_marker == 2:  # SMC Block
                        data = self._parse_smc_block(data_offset)
                    elif type_marker == 3:  # Regime State
                        data = self._parse_regime_state(data_offset)
                    else:
                        data = None
                    
                    if data:
                        # Publish to MessageBus without GIL blocking
                        self._msgbus.publish(topic="data.rust_bridge", msg=data)
                    
                    self._last_offset = current_offset
                    
            except Exception as e:
                log.error(f"Error in SHM poll loop: {e}")
            
            await asyncio.sleep(poll_interval)

    async def _subscribe(self, topics: list[str]) -> None:
        """Start the SHM polling loop."""
        await self._connect()
        asyncio.create_task(self._poll_shm_loop())
        log.info(f"Subscribed to Rust bridge topics: {topics}")

    async def _unsubscribe(self, topics: list[str]) -> None:
        """Stop the SHM polling loop."""
        await self._disconnect()
        log.info(f"Unsubscribed from Rust bridge topics: {topics}")
