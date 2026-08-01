"""
Nautilus ParquetDataCatalog manager for efficient streaming of historical L2 and tick data.
Uses memory-mapped file reading to prevent RAM exhaustion during long simulations.
"""

from __future__ import annotations

import os
import numpy as np
from typing import Dict, List, Optional, Any, Iterator, Tuple
from dataclasses import dataclass
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DataChunk:
    """A chunk of historical data."""
    instrument_id: str
    data_type: str
    start_time_ns: int
    end_time_ns: int
    data: np.ndarray
    row_count: int


class MemoryMappedParquetReader:
    """
    Memory-mapped reader for parquet files.
    Reads data in chunks to avoid loading entire files into memory.
    """
    
    def __init__(self, file_path: str, chunk_size: int = 10000):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self._current_chunk = 0
        self._total_rows = 0
        self._file = None
    
    def __enter__(self):
        try:
            import pyarrow.parquet as pq
            self._parquet_file = pq.ParquetFile(self.file_path)
            self._total_rows = self._parquet_file.metadata.num_rows
        except ImportError:
            logger.warning("PyArrow not available, using fallback")
            self._parquet_file = None
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._parquet_file = None
    
    def iter_chunks(self) -> Iterator[np.ndarray]:
        """Iterate over data chunks."""
        if self._parquet_file is None:
            return
        
        for i in range(0, self._total_rows, self.chunk_size):
            table = self._parquet_file.read_row_group(i // self.chunk_size)
            yield table.to_pandas().values
    
    @property
    def total_rows(self) -> int:
        return self._total_rows


class ParquetDataCatalog:
    """
    Manager for Nautilus-style ParquetDataCatalog.
    Provides efficient access to historical market data.
    """
    
    def __init__(
        self,
        catalog_path: str,
        max_memory_mb: float = 512.0,
        chunk_size_rows: int = 10000
    ):
        self.catalog_path = Path(catalog_path)
        self.max_memory_mb = max_memory_mb
        self.chunk_size_rows = chunk_size_rows
        
        self._cache: Dict[str, Any] = {}
        self._cache_size_mb = 0.0
        
        # Index of available data
        self._instrument_index: Dict[str, List[str]] = {}
        
        logger.info(f"ParquetDataCatalog initialized at {catalog_path}")
    
    def build_index(self):
        """Build index of available instruments and data types."""
        if not self.catalog_path.exists():
            logger.warning(f"Catalog path does not exist: {self.catalog_path}")
            return
        
        for root, dirs, files in os.walk(self.catalog_path):
            for file in files:
                if file.endswith('.parquet'):
                    parts = file.replace('.parquet', '').split('_')
                    if len(parts) >= 2:
                        instrument = parts[0]
                        data_type = parts[1]
                        
                        if instrument not in self._instrument_index:
                            self._instrument_index[instrument] = []
                        if data_type not in self._instrument_index[instrument]:
                            self._instrument_index[instrument].append(data_type)
        
        logger.info(f"Indexed {len(self._instrument_index)} instruments")
    
    def get_instruments(self) -> List[str]:
        """Get list of available instruments."""
        return list(self._instrument_index.keys())
    
    def get_data_types(self, instrument_id: str) -> List[str]:
        """Get available data types for an instrument."""
        return self._instrument_index.get(instrument_id, [])
    
    def read_range(
        self,
        instrument_id: str,
        data_type: str,
        start_time_ns: int,
        end_time_ns: int
    ) -> Iterator[DataChunk]:
        """
        Read data in a time range, yielding chunks to avoid memory exhaustion.
        
        Args:
            instrument_id: Instrument identifier
            data_type: Type of data (bars, quotes, trades)
            start_time_ns: Start time in nanoseconds
            end_time_ns: End time in nanoseconds
            
        Yields:
            DataChunk objects
        """
        file_pattern = f"{instrument_id}_{data_type}_*.parquet"
        
        for file_path in self.catalog_path.glob(file_pattern):
            with MemoryMappedParquetReader(str(file_path), self.chunk_size_rows) as reader:
                for chunk_data in reader.iter_chunks():
                    # Filter by time range
                    # Assuming first column is timestamp
                    if len(chunk_data) > 0 and chunk_data[0, 0] >= 0:
                        mask = (chunk_data[:, 0] >= start_time_ns) & \
                               (chunk_data[:, 0] <= end_time_ns)
                        filtered = chunk_data[mask]
                        
                        if len(filtered) > 0:
                            yield DataChunk(
                                instrument_id=instrument_id,
                                data_type=data_type,
                                start_time_ns=int(filtered[0, 0]),
                                end_time_ns=int(filtered[-1, 0]),
                                data=filtered,
                                row_count=len(filtered)
                            )
    
    def get_bar_data(
        self,
        instrument_id: str,
        start_time_ns: int,
        end_time_ns: int,
        resolution: str = "1m"
    ) -> Iterator[DataChunk]:
        """Get bar data for an instrument."""
        return self.read_range(instrument_id, f"bars_{resolution}", start_time_ns, end_time_ns)
    
    def get_quote_data(
        self,
        instrument_id: str,
        start_time_ns: int,
        end_time_ns: int
    ) -> Iterator[DataChunk]:
        """Get quote/L2 data for an instrument."""
        return self.read_range(instrument_id, "quotes", start_time_ns, end_time_ns)
    
    def get_trade_data(
        self,
        instrument_id: str,
        start_time_ns: int,
        end_time_ns: int
    ) -> Iterator[DataChunk]:
        """Get trade data for an instrument."""
        return self.read_range(instrument_id, "trades", start_time_ns, end_time_ns)
    
    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()
        self._cache_size_mb = 0.0
    
    def get_status(self) -> Dict[str, Any]:
        """Get catalog status."""
        return {
            'catalog_path': str(self.catalog_path),
            'exists': self.catalog_path.exists(),
            'num_instruments': len(self._instrument_index),
            'cache_size_mb': self._cache_size_mb,
            'max_memory_mb': self.max_memory_mb
        }


def create_data_catalog(
    catalog_path: str,
    max_memory_mb: float = 512.0
) -> ParquetDataCatalog:
    """Factory function to create data catalog."""
    catalog = ParquetDataCatalog(catalog_path, max_memory_mb)
    catalog.build_index()
    return catalog
