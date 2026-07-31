//! IPC Contracts Module Root
//!
//! This module exports all shared memory schemas and signal structures
//! for the Python FFI generator and cross-language communication.
//!
//! The contracts defined here enable zero-copy data exchange between
//! the Rust HFT core and Python/Nautilus/Ray ML backend.

pub mod schema;
pub mod signals;

// Re-export primary types for convenience
pub use schema::{
    FeatureElement,
    FeatureVectorHeader,
    FeatureVectorSegment,
    FeatureRingBufferState,
    SHM_MAGIC,
    SCHEMA_VERSION,
};

pub use signals::{
    SignalType,
    AlphaSignal,
    SignalBatchHeader,
    SignalBatchSegment,
    WeightUpdate,
    WeightUpdateBatch,
    SignalQueueState,
};

/// Contract version for compatibility checking
pub const CONTRACT_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Generate FFI header documentation for Python developers
pub fn generate_ffi_docs() -> String {
    format!(
        r#"# IPC Contracts Documentation (Version {})

## Shared Memory Layout

### Feature Vector Segment (Outbound: Rust -> Python)
- Total Size: {} bytes
- Header: {} bytes
- Features: 256 elements × {} bytes = {} bytes
- Padding: 64 bytes

### Signal Batch Segment (Inbound: Python -> Rust)
- Total Size: {} bytes
- Header: {} bytes
- Signals: 64 elements × {} bytes = {} bytes
- Padding: 64 bytes

## Memory Alignment
All structs use #[repr(C)] with explicit padding to ensure:
- 8-byte alignment for f64 values
- Cache-line alignment (64 bytes) for ring buffer states
- Compatible layout with numpy structured arrays

## Usage in Python
```python
import numpy as np
import ctypes

# Load shared memory segment
feature_dtype = np.dtype([
    ('value_bits', np.uint64),
    ('timestamp_ns', np.uint64),
    ('feature_id', np.uint32),
    ('_padding', np.uint32),
])

# Access as zero-copy numpy array
features = np.frombuffer(shm.buf, dtype=feature_dtype, count=256)
```
"#,
        CONTRACT_VERSION,
        FeatureVectorSegment::SIZE,
        std::mem::size_of::<FeatureVectorHeader>(),
        std::mem::size_of::<FeatureElement>(),
        256 * std::mem::size_of::<FeatureElement>(),
        SignalBatchSegment::SIZE,
        std::mem::size_of::<SignalBatchHeader>(),
        std::mem::size_of::<AlphaSignal>(),
        64 * std::mem::size_of::<AlphaSignal>(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contract_version() {
        assert!(!CONTRACT_VERSION.is_empty());
    }

    #[test]
    fn test_magic_number() {
        assert_eq!(SHM_MAGIC, 0x4846545F53484D00);
    }

    #[test]
    fn test_schema_version() {
        assert_eq!(SCHEMA_VERSION, 1);
    }

    #[test]
    fn test_docs_generation() {
        let docs = generate_ffi_docs();
        assert!(docs.contains("IPC Contracts Documentation"));
        assert!(docs.contains("Feature Vector Segment"));
        assert!(docs.contains("Signal Batch Segment"));
    }
}
