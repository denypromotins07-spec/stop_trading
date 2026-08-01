"""
High-performance normalizer for Kaiko deep order book and trade data formats.
Aligns disparate historical venue timestamps to unified nanosecond epoch.
Enables accurate cross-exchange backtesting with memory-efficient processing.
Uses polars/pyarrow for chunked reads respecting 3GB RAM limit.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterator, Tuple, Any
from pathlib import Path
from datetime import datetime, timezone
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class NormalizedTrade:
    """Normalized trade record with unified timestamp."""
    timestamp_ns: int
    exchange: str
    symbol: str
    price: float
    quantity: float
    side: int  # 1 = buy, -1 = sell, 0 = unknown
    trade_id: Optional[str] = None
    original_timestamp: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp_ns': self.timestamp_ns,
            'exchange': self.exchange,
            'symbol': self.symbol,
            'price': self.price,
            'quantity': self.quantity,
            'side': self.side,
            'trade_id': self.trade_id,
            'original_timestamp': self.original_timestamp,
        }


@dataclass
class NormalizedOrderbookLevel:
    """Single level of normalized order book."""
    price: float
    quantity: float
    order_count: int = 1


@dataclass
class NormalizedOrderbookSnapshot:
    """Normalized L2 order book snapshot."""
    timestamp_ns: int
    exchange: str
    symbol: str
    bids: List[NormalizedOrderbookLevel]
    asks: List[NormalizedOrderbookLevel]
    sequence_number: Optional[int] = None
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price."""
        if not self.bids or not self.asks:
            return 0.0
        return (self.bids[0].price + self.asks[0].price) / 2
    
    @property
    def spread_bps(self) -> float:
        """Calculate spread in basis points."""
        if not self.bids or not self.asks or self.mid_price == 0:
            return 0.0
        return (self.asks[0].price - self.bids[0].price) / self.mid_price * 10000


class TimestampNormalizer:
    """
    Normalizes timestamps from different exchanges to unified nanosecond epoch.
    Handles various timestamp formats and timezone conversions.
    """
    
    # Exchange-specific timestamp offsets (nanoseconds) for clock skew correction
    EXCHANGE_OFFSETS = {
        'binance': 0,
        'coinbase': 50_000_000,  # 50ms estimated latency
        'kraken': 75_000_000,
        'ftx': 0,  # Historical data
        'bybit': 30_000_000,
        'okx': 40_000_000,
    }
    
    @classmethod
    def normalize_timestamp(cls, timestamp: Any, exchange: str,
                           unit: str = 'ns') -> int:
        """
        Normalize timestamp to nanoseconds since epoch UTC.
        
        Args:
            timestamp: Input timestamp (various formats)
            exchange: Exchange name for offset correction
            unit: Input unit ('ns', 'us', 'ms', 's')
            
        Returns:
            Nanoseconds since Unix epoch
        """
        # Convert to nanoseconds based on input unit
        if isinstance(timestamp, (int, float)):
            ts_ns = int(timestamp)
            
            if unit == 's':
                ts_ns *= 1_000_000_000
            elif unit == 'ms':
                ts_ns *= 1_000_000
            elif unit == 'us':
                ts_ns *= 1_000
            # Already in ns
            
        elif isinstance(timestamp, datetime):
            # Convert datetime to nanoseconds
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            ts_ns = int(timestamp.timestamp() * 1_000_000_000)
        else:
            logger.warning(f"Unknown timestamp format: {type(timestamp)}")
            ts_ns = 0
        
        # Apply exchange-specific offset for clock skew correction
        offset = cls.EXCHANGE_OFFSETS.get(exchange.lower(), 0)
        ts_ns += offset
        
        return ts_ns
    
    @classmethod
    def align_timestamps(cls, timestamps: np.ndarray, 
                        exchanges: List[str],
                        target_resolution_ns: int = 1_000_000  # 1ms
                        ) -> np.ndarray:
        """
        Align multiple timestamps to common grid.
        
        Args:
            timestamps: Array of timestamps in nanoseconds
            exchanges: List of exchange names for each timestamp
            target_resolution_ns: Target time grid resolution
            
        Returns:
            Aligned timestamps
        """
        aligned = np.zeros_like(timestamps)
        
        for i, (ts, exchange) in enumerate(zip(timestamps, exchanges)):
            # Apply offset
            offset = cls.EXCHANGE_OFFSETS.get(exchange.lower(), 0)
            adjusted = ts + offset
            
            # Round to grid
            aligned[i] = (adjusted // target_resolution_ns) * target_resolution_ns
        
        return aligned


class KaikoNormalizer:
    """
    High-performance normalizer for Kaiko data formats.
    
    Features:
    - Chunked reading with polars/pyarrow
    - Unified timestamp normalization
    - Memory-bounded processing
    - Cross-exchange alignment
    """
    
    def __init__(self, 
                 output_dir: Optional[Path] = None,
                 max_memory_rows: int = 100000,
                 chunk_size: int = 10000):
        """
        Initialize normalizer.
        
        Args:
            output_dir: Output directory for normalized data
            max_memory_rows: Maximum rows to hold in memory
            chunk_size: Processing chunk size
        """
        self._output_dir = output_dir or Path("./data/kaiko_normalized")
        self._max_memory_rows = max_memory_rows
        self._chunk_size = chunk_size
        self._timestamp_normalizer = TimestampNormalizer()
        
        # Statistics
        self._stats = {
            'files_processed': 0,
            'trades_normalized': 0,
            'snapshots_normalized': 0,
            'timestamp_corrections': 0,
        }
        
        self._output_dir.mkdir(parents=True, exist_ok=True)
    
    def normalize_trade_file(self, input_path: Path,
                            exchange: str,
                            symbol: str,
                            output_path: Optional[Path] = None
                            ) -> Iterator[NormalizedTrade]:
        """
        Normalize trades from Kaiko format.
        
        Args:
            input_path: Input file path (parquet/csv)
            exchange: Exchange name
            symbol: Symbol
            output_path: Optional output path for normalized data
            
        Yields:
            NormalizedTrade objects (streaming)
        """
        if output_path is None:
            output_path = self._output_dir / f"{exchange}_{symbol}_trades.parquet"
        
        # Read in chunks using pyarrow
        parquet_file = pq.ParquetFile(input_path)
        
        writer = None
        schema = self._get_trade_schema()
        
        for batch in parquet_file.iter_batches(batch_size=self._chunk_size):
            table = pa.Table.from_batches([batch])
            
            # Process chunk
            trades = self._normalize_trade_chunk(
                table, exchange, symbol
            )
            
            # Write to output
            if writer is None:
                writer = pq.ParquetWriter(output_path, schema, compression='snappy')
            
            # Convert trades to table
            trade_table = self._trades_to_table(trades, schema)
            if trade_table.num_rows > 0:
                writer.write_table(trade_table)
                self._stats['trades_normalized'] += trade_table.num_rows
            
            # Yield for streaming consumption
            for trade in trades:
                yield trade
        
        if writer:
            writer.close()
        
        self._stats['files_processed'] += 1
    
    def _normalize_trade_chunk(self, table: pa.Table,
                               exchange: str,
                               symbol: str) -> List[NormalizedTrade]:
        """Normalize a chunk of trades."""
        trades = []
        
        # Extract columns
        timestamps = table.column('timestamp').to_numpy()
        prices = table.column('price').to_numpy()
        quantities = table.column('amount').to_numpy()
        
        # Determine side column (may vary by format)
        side_col = None
        for col_name in ['side', 'taker_side', 'type']:
            if col_name in table.column_names:
                side_col = table.column(col_name).to_numpy()
                break
        
        # Trade IDs (optional)
        trade_ids = None
        if 'id' in table.column_names or 'trade_id' in table.column_names:
            id_col = 'id' if 'id' in table.column_names else 'trade_id'
            trade_ids = table.column(id_col).to_numpy()
        
        # Normalize timestamps
        ts_unit = self._infer_timestamp_unit(timestamps)
        normalized_ts = np.array([
            self._timestamp_normalizer.normalize_timestamp(ts, exchange, ts_unit)
            for ts in timestamps
        ])
        
        # Track corrections
        if not np.array_equal(timestamps, normalized_ts):
            self._stats['timestamp_corrections'] += len(normalized_ts)
        
        # Create normalized trades
        for i in range(len(table)):
            side = 0
            if side_col is not None:
                side_val = side_col[i]
                if isinstance(side_val, str):
                    side = 1 if side_val.lower() in ['buy', 'bid'] else -1
                else:
                    side = 1 if side_val > 0 else -1
            
            trade = NormalizedTrade(
                timestamp_ns=int(normalized_ts[i]),
                exchange=exchange,
                symbol=symbol,
                price=float(prices[i]),
                quantity=float(quantities[i]),
                side=side,
                trade_id=str(trade_ids[i]) if trade_ids is not None else None,
                original_timestamp=int(timestamps[i]),
            )
            trades.append(trade)
        
        return trades
    
    def normalize_orderbook_file(self, input_path: Path,
                                 exchange: str,
                                 symbol: str,
                                 output_path: Optional[Path] = None
                                 ) -> Iterator[NormalizedOrderbookSnapshot]:
        """
        Normalize order book snapshots from Kaiko format.
        
        Args:
            input_path: Input file path
            exchange: Exchange name
            symbol: Symbol
            output_path: Optional output path
            
        Yields:
            NormalizedOrderbookSnapshot objects
        """
        if output_path is None:
            output_path = self._output_dir / f"{exchange}_{symbol}_orderbook.parquet"
        
        parquet_file = pq.ParquetFile(input_path)
        
        for batch in parquet_file.iter_batches(batch_size=self._chunk_size):
            table = pa.Table.from_batches([batch])
            snapshots = self._normalize_orderbook_chunk(table, exchange, symbol)
            
            for snapshot in snapshots:
                yield snapshot
        
        self._stats['files_processed'] += 1
    
    def _normalize_orderbook_chunk(self, table: pa.Table,
                                   exchange: str,
                                   symbol: str
                                   ) -> List[NormalizedOrderbookSnapshot]:
        """Normalize a chunk of order book snapshots."""
        snapshots = []
        
        timestamps = table.column('timestamp').to_numpy()
        ts_unit = self._infer_timestamp_unit(timestamps)
        
        # Get bid/ask columns (format varies)
        bid_cols = [c for c in table.column_names if c.startswith('b')]
        ask_cols = [c for c in table.column_names if c.startswith('a')]
        
        for i in range(len(table)):
            ts_ns = self._timestamp_normalizer.normalize_timestamp(
                timestamps[i], exchange, ts_unit
            )
            
            # Extract bids
            bids = []
            for j in range(25):  # Up to 25 levels
                price_col = f'bid_price_{j}'
                qty_col = f'bid_amount_{j}'
                
                if price_col in table.column_names and qty_col in table.column_names:
                    price = table.column(price_col)[i].as_py()
                    qty = table.column(qty_col)[i].as_py()
                    
                    if price is not None and price > 0:
                        bids.append(NormalizedOrderbookLevel(
                            price=float(price),
                            quantity=float(qty),
                        ))
            
            # Extract asks
            asks = []
            for j in range(25):
                price_col = f'ask_price_{j}'
                qty_col = f'ask_amount_{j}'
                
                if price_col in table.column_names and qty_col in table.column_names:
                    price = table.column(price_col)[i].as_py()
                    qty = table.column(qty_col)[i].as_py()
                    
                    if price is not None and price > 0:
                        asks.append(NormalizedOrderbookLevel(
                            price=float(price),
                            quantity=float(qty),
                        ))
            
            snapshot = NormalizedOrderbookSnapshot(
                timestamp_ns=ts_ns,
                exchange=exchange,
                symbol=symbol,
                bids=bids,
                asks=asks,
            )
            snapshots.append(snapshot)
        
        self._stats['snapshots_normalized'] += len(snapshots)
        return snapshots
    
    def _infer_timestamp_unit(self, timestamps: np.ndarray) -> str:
        """Infer timestamp unit from magnitude."""
        if len(timestamps) == 0:
            return 'ns'
        
        sample = abs(timestamps[0])
        
        if sample > 1e18:
            return 'ns'
        elif sample > 1e15:
            return 'us'
        elif sample > 1e12:
            return 'ms'
        else:
            return 's'
    
    def _get_trade_schema(self) -> pa.Schema:
        """Get pyarrow schema for normalized trades."""
        return pa.schema([
            ('timestamp_ns', pa.int64()),
            ('exchange', pa.string()),
            ('symbol', pa.string()),
            ('price', pa.float64()),
            ('quantity', pa.float64()),
            ('side', pa.int8()),
            ('trade_id', pa.string()),
            ('original_timestamp', pa.int64()),
        ])
    
    def _trades_to_table(self, trades: List[NormalizedTrade],
                         schema: pa.Schema) -> pa.Table:
        """Convert list of trades to pyarrow Table."""
        if not trades:
            return pa.Table.from_arrays([[]] * len(schema), schema=schema)
        
        arrays = {
            'timestamp_ns': [t.timestamp_ns for t in trades],
            'exchange': [t.exchange for t in trades],
            'symbol': [t.symbol for t in trades],
            'price': [t.price for t in trades],
            'quantity': [t.quantity for t in trades],
            'side': [t.side for t in trades],
            'trade_id': [t.trade_id or '' for t in trades],
            'original_timestamp': [t.original_timestamp or 0 for t in trades],
        }
        
        return pa.Table.from_pydict(arrays, schema=schema)
    
    def merge_exchanges(self, normalized_files: List[Path],
                       output_path: Path,
                       sort_by_time: bool = True) -> Path:
        """
        Merge multiple normalized files into single dataset.
        
        Args:
            normalized_files: List of normalized file paths
            output_path: Output merged file path
            sort_by_time: Sort by timestamp
            
        Returns:
            Path to merged file
        """
        tables = []
        
        for file_path in normalized_files:
            pf = pq.ParquetFile(file_path)
            for batch in pf.iter_batches(batch_size=self._chunk_size):
                tables.append(pa.Table.from_batches([batch]))
        
        if not tables:
            raise ValueError("No tables to merge")
        
        # Concatenate all tables
        merged = pa.concat_tables(tables)
        
        if sort_by_time:
            # Sort by timestamp_ns
            indices = pa.compute.sort_indices(merged, sort_keys=[('timestamp_ns', 'ascending')])
            merged = merged.take(indices)
        
        # Write merged file
        pq.write_table(merged, output_path, compression='snappy')
        
        logger.info(f"Merged {len(normalized_files)} files to {output_path}")
        return output_path
    
    def get_stats(self) -> Dict:
        """Get normalization statistics."""
        return self._stats.copy()
    
    def validate_alignment(self, trades: List[NormalizedTrade],
                          max_drift_ms: float = 100.0) -> Dict[str, Any]:
        """
        Validate timestamp alignment across exchanges.
        
        Args:
            trades: List of normalized trades
            max_drift_ms: Maximum acceptable drift in milliseconds
            
        Returns:
            Validation results
        """
        if len(trades) < 2:
            return {'valid': True, 'max_drift_ns': 0}
        
        # Group by exchange
        by_exchange: Dict[str, List[int]] = {}
        for trade in trades:
            if trade.exchange not in by_exchange:
                by_exchange[trade.exchange] = []
            by_exchange[trade.exchange].append(trade.timestamp_ns)
        
        # Check for simultaneous trades across exchanges
        max_drift = 0
        exchanges = list(by_exchange.keys())
        
        for i in range(len(exchanges)):
            for j in range(i + 1, len(exchanges)):
                ex1, ex2 = exchanges[i], exchanges[j]
                ts1, ts2 = by_exchange[ex1], by_exchange[ex2]
                
                # Compare median timestamps
                median1 = np.median(ts1)
                median2 = np.median(ts2)
                drift = abs(median1 - median2)
                max_drift = max(max_drift, drift)
        
        drift_ms = max_drift / 1_000_000
        
        return {
            'valid': drift_ms <= max_drift_ms,
            'max_drift_ns': max_drift,
            'max_drift_ms': drift_ms,
            'exchanges_compared': len(exchanges),
        }
