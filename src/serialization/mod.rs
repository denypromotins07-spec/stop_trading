//! Serialization Module Root
//!
//! Provides trait abstractions for pluggable serialization formats.
//! Integrates rkyv for zero-copy deserialization and FlatBuffers
//! for cross-language IPC with Python/Ray Nautilus bridge.

pub mod rkyv_impl;
pub mod flatbuffers;

pub use rkyv_impl::{RkyvSerializer, RkyvDeserializer, ArchivedState};
pub use flatbuffers::{FlatBufferBuilder, SchemaRegistry, FeatureVector};

use std::sync::atomic::{AtomicU64, Ordering};

/// Serialization format trait
pub trait Serializer<T>: Send + Sync {
    /// Serialize data to bytes
    fn serialize(&self, data: &T) -> Result<Vec<u8>, SerializationError>;
    
    /// Get serialized size
    fn serialized_size(&self, data: &T) -> usize;
}

/// Deserialization format trait
pub trait Deserializer<T>: Send + Sync {
    /// Deserialize from bytes
    fn deserialize<'a>(&self, bytes: &'a [u8]) -> Result<T, SerializationError>;
    
    /// Zero-copy deserialize (if supported)
    fn deserialize_zero_copy<'a>(&self, bytes: &'a [u8]) -> Result<&'a T, SerializationError>;
}

/// Serialization error types
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerializationError {
    /// Buffer too small
    BufferTooSmall,
    /// Invalid format
    InvalidFormat,
    /// Missing required field
    MissingField,
    /// Type mismatch
    TypeMismatch,
    /// Alignment error
    AlignmentError,
    /// Out of bounds
    OutOfBounds,
    /// Unsupported operation
    Unsupported,
}

impl SerializationError {
    #[inline]
    pub fn error_code(&self) -> u32 {
        match self {
            SerializationError::BufferTooSmall => 1,
            SerializationError::InvalidFormat => 2,
            SerializationError::MissingField => 3,
            SerializationError::TypeMismatch => 4,
            SerializationError::AlignmentError => 5,
            SerializationError::OutOfBounds => 6,
            SerializationError::Unsupported => 7,
        }
    }
}

/// Pluggable serialization manager
#[repr(C)]
pub struct SerializationManager {
    /// Total serializations performed
    serializations: AtomicU64,
    /// Total deserializations performed
    deserializations: AtomicU64,
    /// Bytes processed
    bytes_processed: AtomicU64,
    /// Serialization errors
    errors: AtomicU64,
}

impl SerializationManager {
    pub fn new() -> Self {
        Self {
            serializations: AtomicU64::new(0),
            deserializations: AtomicU64::new(0),
            bytes_processed: AtomicU64::new(0),
            errors: AtomicU64::new(0),
        }
    }

    /// Record a serialization operation
    #[inline]
    pub fn record_serialization(&self, bytes: usize) {
        self.serializations.fetch_add(1, Ordering::Relaxed);
        self.bytes_processed.fetch_add(bytes as u64, Ordering::Relaxed);
    }

    /// Record a deserialization operation
    #[inline]
    pub fn record_deserialization(&self, bytes: usize) {
        self.deserializations.fetch_add(1, Ordering::Relaxed);
        self.bytes_processed.fetch_add(bytes as u64, Ordering::Relaxed);
    }

    /// Record an error
    #[inline]
    pub fn record_error(&self) {
        self.errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Get statistics
    #[inline]
    pub fn get_stats(&self) -> SerializationStats {
        SerializationStats {
            serializations: self.serializations.load(Ordering::Relaxed),
            deserializations: self.deserializations.load(Ordering::Relaxed),
            bytes_processed: self.bytes_processed.load(Ordering::Relaxed),
            errors: self.errors.load(Ordering::Relaxed),
        }
    }
}

impl Default for SerializationManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Serialization statistics
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SerializationStats {
    pub serializations: u64,
    pub deserializations: u64,
    pub bytes_processed: u64,
    pub errors: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialization_manager() {
        let manager = SerializationManager::new();
        
        assert_eq!(manager.get_stats().serializations, 0);
        assert_eq!(manager.get_stats().deserializations, 0);
        
        manager.record_serialization(100);
        manager.record_deserialization(100);
        
        let stats = manager.get_stats();
        assert_eq!(stats.serializations, 1);
        assert_eq!(stats.deserializations, 1);
        assert_eq!(stats.bytes_processed, 200);
    }

    #[test]
    fn test_error_codes() {
        assert_eq!(SerializationError::BufferTooSmall.error_code(), 1);
        assert_eq!(SerializationError::InvalidFormat.error_code(), 2);
        assert_eq!(SerializationError::Unsupported.error_code(), 7);
    }
}
