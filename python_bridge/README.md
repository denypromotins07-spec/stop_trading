# Python Bridge - IPC Interface for Nautilus/Ray ML Backend

## Overview

This directory contains the Python ML backend that communicates with the Rust HFT core via shared memory IPC. The interface enables zero-copy data exchange for ultra-low latency feature vectors and alpha signals.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HFT Crypto Bot System                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    Shared Memory    ┌──────────────┐          │
│  │   Rust Core  │◄───────────────────►│  Python ML   │          │
│  │              │                     │   Backend    │          │
│  │ - Disruptor  │  Feature Vectors    │ - Nautilus   │          │
│  │ - Order Book │  (Outbound)         │ - Ray        │          │
│  │ - Execution  │                     │ - PyTorch    │          │
│  └──────────────┘  Alpha Signals      └──────────────┘          │
│                      (Inbound)                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Memory Layout Specifications

### Feature Vector Segment (Rust → Python)

**Total Size:** 6,304 bytes  
**Alignment:** 8-byte aligned

| Offset | Field | Type | Size (bytes) | Description |
|--------|-------|------|--------------|-------------|
| 0 | magic | u64 | 8 | Magic number: 0x4846545F53484D00 |
| 8 | version | u32 | 4 | Schema version |
| 12 | _pad | u32 | 4 | Padding |
| 16 | sequence | u64 | 8 | Monotonic sequence number |
| 24 | symbol | u64 | 8 | Symbol identifier |
| 32 | timestamp_ns | u64 | 8 | Generation timestamp |
| 40 | feature_count | u32 | 4 | Number of valid features |
| 44 | vector_id | u32 | 4 | Vector identifier |
| 48 | checksum | u32 | 4 | CRC32 checksum |
| 52 | flags | u32 | 4 | Status flags |
| 56 | _reserved | u64[4] | 32 | Reserved |
| 88 | features[0] | FeatureElement | 24 | First feature |
| ... | features[n] | FeatureElement | 24 | Feature n |
| 6240 | _padding | u8[64] | 64 | Cache line padding |

**FeatureElement Structure (24 bytes):**
| Offset | Field | Type | Description |
|--------|-------|------|-------------|
| 0 | value_bits | u64 | f64 as bits |
| 8 | timestamp_ns | u64 | Feature timestamp |
| 16 | feature_id | u32 | Feature identifier |
| 20 | _padding | u32 | Alignment padding |

### Signal Batch Segment (Python → Rust)

**Total Size:** 3,232 bytes  
**Alignment:** 8-byte aligned

| Offset | Field | Type | Size (bytes) | Description |
|--------|-------|------|--------------|-------------|
| 0 | header | SignalBatchHeader | 96 | Batch metadata |
| 96 | signals[0] | AlphaSignal | 48 | First signal |
| ... | signals[n] | AlphaSignal | 48 | Signal n |
| 3168 | _padding | u8[64] | 64 | Cache line padding |

**AlphaSignal Structure (48 bytes):**
| Offset | Field | Type | Description |
|--------|-------|------|-------------|
| 0 | signal_type | u8 | Signal type enum |
| 1 | confidence | u8 | Confidence 0-100 |
| 2-3 | _padding | u8[2] | Alignment |
| 4 | symbol | u64 | Target symbol |
| 12 | value | i64 | Fixed-point value (×1e9) |
| 20 | timestamp_ns | u64 | Signal timestamp |
| 28 | expiry_ns | u64 | Expiry timestamp |
| 36 | model_id | u32 | Model identifier |
| 40 | sequence | u64 | Sequence number |

## FFI Function Signatures

The Python backend must implement these entry points:

```python
# ml_backend.py

def initialize(shm_feature_path: str, shm_signal_path: str) -> bool:
    """
    Initialize shared memory mappings.
    
    Args:
        shm_feature_path: Path to feature vector shared memory
        shm_signal_path: Path to signal batch shared memory
    
    Returns:
        True if initialization successful
    """
    pass

def process_features(features: np.ndarray) -> Optional[AlphaSignal]:
    """
    Process incoming feature vector and generate alpha signal.
    
    Args:
        features: Zero-copy numpy array of features
    
    Returns:
        AlphaSignal or None if no trade opportunity
    """
    pass

def shutdown() -> None:
    """Clean shutdown, release shared memory."""
    pass
```

## Usage Example

```python
#!/usr/bin/env python3
"""Example ML Backend Implementation"""

import numpy as np
from nautilus_trader.core import DataEngine
from ray.util.actor_pool import ActorPool
import mmap
from typing import Optional

from hft_bridge import AlphaSignal, SignalType, FeatureVectorSegment

class MLBackend:
    def __init__(self):
        self.feature_shm = None
        self.signal_shm = None
        self.features_view = None
        self.signals_view = None
        
    def initialize(self, feature_path: str, signal_path: str) -> bool:
        # Map shared memory regions
        self.feature_shm = mmap.mmap(-1, 6304, tagname=feature_path)
        self.signal_shm = mmap.mmap(-1, 3232, tagname=signal_path)
        
        # Create numpy views for zero-copy access
        self.features_view = np.frombuffer(
            self.feature_shm, 
            dtype=np.dtype([
                ('header', 'u8', 12),
                ('features', 'u8', 256*24)
            ])
        )
        return True
    
    def run(self):
        """Main processing loop"""
        while True:
            # Check for new features
            if self._has_new_features():
                features = self._read_features()
                signal = self._compute_alpha(features)
                if signal:
                    self._write_signal(signal)
    
    def _compute_alpha(self, features: np.ndarray) -> Optional[AlphaSignal]:
        """ML inference to compute directional alpha"""
        # Implement your strategy here
        pass
```

## Memory Safety Requirements

1. **No allocations during hot path**: All buffers pre-allocated
2. **Lock-free coordination**: Use atomic sequence numbers
3. **Cache-line alignment**: Prevent false sharing
4. **Checksum validation**: Verify data integrity on read

## Testing

```bash
# Run integration tests
pytest tests/test_ipc_bridge.py -v

# Benchmark latency
python benchmarks/latency_test.py

# Validate memory layout
python tools/validate_schema.py
```

## Performance Targets

| Operation | Target Latency |
|-----------|----------------|
| Feature read (zero-copy) | < 100 ns |
| Signal generation | < 10 μs |
| Signal write | < 500 ns |
| Round-trip IPC | < 5 μs |

## Troubleshooting

### Common Issues

1. **Schema mismatch**: Verify `SCHEMA_VERSION` matches between Rust and Python
2. **Alignment errors**: Ensure numpy dtype matches C struct exactly
3. **Memory corruption**: Check checksums and magic numbers
4. **Stale data**: Monitor sequence numbers for gaps

### Debug Tools

```bash
# Inspect shared memory
./tools/dump_shm.py

# Validate schema alignment
./tools/validate_layout.py

# Latency profiler
./tools/profile_ipc.py
```
