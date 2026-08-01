"""
Module root streaming on-chain transaction logs from Rust into the Python graph analytics engine.
Uses bounded queues for memory-safe IPC communication with the Rust core.
"""

from __future__ import annotations

import asyncio
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import logging
import time
from enum import Enum
import struct

logger = logging.getLogger(__name__)


class TransactionType(Enum):
    """Types of on-chain transactions."""
    TRANSFER = "transfer"
    SWAP = "swap"
    LIQUIDITY_ADD = "liquidity_add"
    LIQUIDITY_REMOVE = "liquidity_remove"
    CONTRACT_CALL = "contract_call"
    MEV_ARBITRAGE = "mev_arbitrage"
    MEV_LIQUIDATION = "mev_liquidation"
    UNKNOWN = "unknown"


@dataclass
class OnChainTransaction:
    """Parsed on-chain transaction record."""
    tx_hash: str
    block_number: int
    timestamp_ns: int
    
    from_address: str
    to_address: str
    value_eth: float
    gas_used: int
    gas_price_gwei: float
    
    tx_type: TransactionType = TransactionType.UNKNOWN
    
    # Additional metadata
    contract_address: Optional[str] = None
    method_id: Optional[str] = None
    input_data: Optional[bytes] = None
    
    # Parsed fields
    token_transfers: List[Tuple[str, str, float]] = field(default_factory=list)
    
    def to_graph_input(self) -> Tuple[str, str, float]:
        """Convert to graph input format."""
        return (self.from_address, self.to_address, self.value_eth)


class BoundedTransactionQueue:
    """
    Thread-safe bounded queue for transaction streaming.
    Prevents memory exhaustion by dropping old transactions when full.
    """
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._queue: deque = deque()
        self._lock = asyncio.Lock()
        self._dropped_count = 0
        self._processed_count = 0
    
    @property
    def size(self) -> int:
        return len(self._queue)
    
    @property
    def dropped_count(self) -> int:
        return self._dropped_count
    
    @property
    def processed_count(self) -> int:
        return self._processed_count
    
    async def put(self, tx: OnChainTransaction) -> bool:
        """Add transaction to queue. Returns False if dropped."""
        async with self._lock:
            if len(self._queue) >= self.max_size:
                # Drop oldest transaction
                self._queue.popleft()
                self._dropped_count += 1
            
            self._queue.append(tx)
            return True
    
    async def get(self) -> Optional[OnChainTransaction]:
        """Get next transaction from queue."""
        async with self._lock:
            if not self._queue:
                return None
            
            tx = self._queue.popleft()
            self._processed_count += 1
            return tx
    
    async def get_batch(self, max_batch: int = 100) -> List[OnChainTransaction]:
        """Get batch of transactions."""
        async with self._lock:
            batch = []
            while len(batch) < max_batch and self._queue:
                batch.append(self._queue.popleft())
                self._processed_count += 1
            return batch
    
    async def clear(self) -> int:
        """Clear queue and return count of cleared items."""
        async with self._lock:
            count = len(self._queue)
            self._queue.clear()
            return count


class TransactionParser:
    """
    Parser for raw on-chain transaction data from Rust IPC.
    Handles various transaction formats and extracts graph-relevant features.
    """
    
    # Known method signatures
    METHOD_SIGNATURES = {
        '0xa9059cbb': 'transfer',
        '0x23b872dd': 'transferFrom',
        '0x095ea7b3': 'approve',
        '0x18cbafe5': 'swapExactTokensForTokens',
        '0x38ed1739': 'swapExactTokensForTokensSupportingFeeOnTransferTokens',
        '0xe8e33700': 'addLiquidity',
        '0xf305d719': 'removeLiquidity',
    }
    
    @classmethod
    def parse_json(cls, json_data: str) -> Optional[OnChainTransaction]:
        """Parse JSON-formatted transaction data."""
        try:
            data = json.loads(json_data)
            
            tx_type = cls._detect_tx_type(data)
            
            return OnChainTransaction(
                tx_hash=data.get('hash', ''),
                block_number=int(data.get('blockNumber', 0)),
                timestamp_ns=int(data.get('timestamp', 0)) * 1_000_000_000,
                from_address=data.get('from', '').lower(),
                to_address=data.get('to', '').lower(),
                value_eth=float(data.get('value', 0)) / 1e18,  # Convert wei to ETH
                gas_used=int(data.get('gasUsed', 0)),
                gas_price_gwei=float(data.get('gasPrice', 0)) / 1e9,
                tx_type=tx_type,
                contract_address=data.get('contractAddress'),
                method_id=data.get('methodId'),
                token_transfers=cls._parse_token_transfers(data)
            )
        except Exception as e:
            logger.error(f"Failed to parse transaction JSON: {e}")
            return None
    
    @classmethod
    def parse_binary(cls, binary_data: bytes) -> Optional[OnChainTransaction]:
        """
        Parse binary-encoded transaction data.
        
        Binary format (little-endian):
        - tx_hash: 32 bytes
        - block_number: 8 bytes (uint64)
        - timestamp: 8 bytes (uint64)
        - from_addr: 20 bytes
        - to_addr: 20 bytes
        - value: 32 bytes (uint256, wei)
        - gas_used: 8 bytes (uint64)
        - gas_price: 8 bytes (uint64, wei)
        """
        try:
            expected_size = 32 + 8 + 8 + 20 + 20 + 32 + 8 + 8
            if len(binary_data) < expected_size:
                raise ValueError(f"Binary data too short: {len(binary_data)} < {expected_size}")
            
            offset = 0
            
            # Parse fields
            tx_hash = binary_data[offset:offset+32].hex()
            offset += 32
            
            block_number = struct.unpack('<Q', binary_data[offset:offset+8])[0]
            offset += 8
            
            timestamp = struct.unpack('<Q', binary_data[offset:offset+8])[0]
            offset += 8
            
            from_addr = '0x' + binary_data[offset:offset+20].hex()
            offset += 20
            
            to_addr = '0x' + binary_data[offset:offset+20].hex()
            offset += 20
            
            value_wei = struct.unpack('<Q', binary_data[offset:offset+8])[0]
            offset += 8
            
            gas_used = struct.unpack('<Q', binary_data[offset:offset+8])[0]
            offset += 8
            
            gas_price = struct.unpack('<Q', binary_data[offset:offset+8])[0]
            
            return OnChainTransaction(
                tx_hash=tx_hash,
                block_number=block_number,
                timestamp_ns=timestamp * 1_000_000_000,
                from_address=from_addr.lower(),
                to_address=to_addr.lower(),
                value_eth=value_wei / 1e18,
                gas_used=gas_used,
                gas_price_gwei=gas_price / 1e9,
                tx_type=TransactionType.TRANSFER
            )
        except Exception as e:
            logger.error(f"Failed to parse binary transaction: {e}")
            return None
    
    @staticmethod
    def _detect_tx_type(data: Dict) -> TransactionType:
        """Detect transaction type from data."""
        method_id = data.get('methodId', '')
        
        if method_id in ['0x18cbafe5', '0x38ed1739']:
            return TransactionType.SWAP
        elif method_id in ['0xe8e33700']:
            return TransactionType.LIQUIDITY_ADD
        elif method_id in ['0xf305d719']:
            return TransactionType.LIQUIDITY_REMOVE
        
        # Check for MEV patterns
        if data.get('isMev', False):
            if 'liquidation' in str(data.get('logs', [])).lower():
                return TransactionType.MEV_LIQUIDATION
            return TransactionType.MEV_ARBITRAGE
        
        # Check for contract interaction
        if data.get('contractAddress'):
            return TransactionType.CONTRACT_CALL
        
        return TransactionType.TRANSFER
    
    @staticmethod
    def _parse_token_transfers(data: Dict) -> List[Tuple[str, str, float]]:
        """Parse token transfer events from transaction logs."""
        transfers = []
        
        logs = data.get('logs', [])
        for log in logs:
            if isinstance(log, dict):
                topics = log.get('topics', [])
                if len(topics) >= 3 and topics[0] == '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef':
                    # ERC20 Transfer event
                    from_addr = '0x' + topics[1][-40:] if len(topics[1]) >= 40 else ''
                    to_addr = '0x' + topics[2][-40:] if len(topics[2]) >= 40 else ''
                    
                    # Parse value from data field
                    value_hex = log.get('data', '0x0')
                    try:
                        value = int(value_hex, 16) / 1e18
                    except:
                        value = 0.0
                    
                    token_addr = log.get('address', '')
                    transfers.append((token_addr, from_addr.lower(), to_addr.lower(), value))
        
        return transfers


class OnChainGraphModule:
    """
    Main module for streaming on-chain data to graph analytics.
    Manages IPC communication with Rust core and feeds the graph engine.
    """
    
    def __init__(
        self,
        queue_max_size: int = 10000,
        batch_size: int = 100,
        graph_analytics: Optional[Any] = None
    ):
        self.transaction_queue = BoundedTransactionQueue(queue_max_size)
        self.parser = TransactionParser()
        self.batch_size = batch_size
        self.graph_analytics = graph_analytics
        
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        self._callbacks: List[Callable[[OnChainTransaction], None]] = []
        
        # Statistics
        self._total_received = 0
        self._total_processed = 0
        self._last_block_seen = 0
        
        logger.info("OnChainGraphModule initialized")
    
    def register_callback(self, callback: Callable[[OnChainTransaction], None]):
        """Register callback for processed transactions."""
        self._callbacks.append(callback)
    
    async def receive_transaction(self, tx: OnChainTransaction) -> bool:
        """Receive a parsed transaction."""
        self._total_received += 1
        self._last_block_seen = max(self._last_block_seen, tx.block_number)
        
        return await self.transaction_queue.put(tx)
    
    async def receive_raw_json(self, json_data: str) -> bool:
        """Receive raw JSON transaction data from Rust IPC."""
        tx = self.parser.parse_json(json_data)
        if tx is None:
            return False
        return await self.receive_transaction(tx)
    
    async def receive_raw_binary(self, binary_data: bytes) -> bool:
        """Receive raw binary transaction data from Rust IPC."""
        tx = self.parser.parse_binary(binary_data)
        if tx is None:
            return False
        return await self.receive_transaction(tx)
    
    async def _process_batch(self, batch: List[OnChainTransaction]):
        """Process a batch of transactions."""
        for tx in batch:
            # Feed to graph analytics
            if self.graph_analytics is not None:
                try:
                    from_addr, to_addr, value = tx.to_graph_input()
                    self.graph_analytics.process_transaction(
                        from_addr, to_addr, value, tx.timestamp_ns
                    )
                except Exception as e:
                    logger.error(f"Error processing transaction for graph: {e}")
            
            # Call registered callbacks
            for callback in self._callbacks:
                try:
                    callback(tx)
                except Exception as e:
                    logger.error(f"Error in transaction callback: {e}")
            
            self._total_processed += 1
    
    async def run_worker(self):
        """Main worker loop processing transactions."""
        self._running = True
        logger.info("OnChain graph worker started")
        
        while self._running:
            try:
                batch = await self.transaction_queue.get_batch(self.batch_size)
                
                if not batch:
                    await asyncio.sleep(0.001)
                    continue
                
                await self._process_batch(batch)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in graph worker: {e}")
        
        self._running = False
        logger.info("OnChain graph worker stopped")
    
    def start(self):
        """Start the async worker."""
        if self._running:
            return
        
        loop = asyncio.get_event_loop()
        self._worker_task = loop.create_task(self.run_worker())
    
    def stop(self):
        """Stop the async worker."""
        self._running = False
        if self._worker_task is not None:
            self._worker_task.cancel()
    
    def get_status(self) -> Dict[str, Any]:
        """Get module status."""
        status = {
            'running': self._running,
            'queue_size': self.transaction_queue.size,
            'dropped_transactions': self.transaction_queue.dropped_count,
            'total_received': self._total_received,
            'total_processed': self._total_processed,
            'last_block_seen': self._last_block_seen
        }
        
        if self.graph_analytics is not None:
            status['graph_stats'] = self.graph_analytics.get_system_status()
        
        return status


# Module singleton
_module_instance: Optional[OnChainGraphModule] = None


def get_module() -> OnChainGraphModule:
    """Get or create module singleton."""
    global _module_instance
    if _module_instance is None:
        _module_instance = OnChainGraphModule()
    return _module_instance


def initialize_module(
    queue_max_size: int = 10000,
    batch_size: int = 100,
    graph_analytics: Optional[Any] = None
) -> OnChainGraphModule:
    """Initialize the module with configuration."""
    global _module_instance
    _module_instance = OnChainGraphModule(
        queue_max_size=queue_max_size,
        batch_size=batch_size,
        graph_analytics=graph_analytics
    )
    return _module_instance
