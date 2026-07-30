//! SIMD Module Root
//! 
//! Abstracts hardware capabilities using safe wrappers over raw `core::arch` intrinsics.

pub mod avx2;
pub mod sse;

pub use avx2::{
    compute_microprice_dispatch, Avx2Accelerator,
};
pub use sse::SseAccelerator;

use std::sync::Arc;
use tracing::{debug, info};

/// Hardware capability detection and feature abstraction
#[derive(Debug, Clone)]
pub struct SimdCapabilities {
    /// SSE4.2 available
    pub sse42: bool,
    /// AVX available
    pub avx: bool,
    /// AVX2 available
    pub avx2: bool,
    /// AVX-512F available
    pub avx512f: bool,
    /// BMI1 (bit manipulation) available
    pub bmi1: bool,
    /// BMI2 available
    pub bmi2: bool,
    /// LZCNT instruction available
    pub lzcnt: bool,
}

impl SimdCapabilities {
    /// Detect current CPU capabilities
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                sse42: is_x86_feature_detected!("sse4.2"),
                avx: is_x86_feature_detected!("avx"),
                avx2: is_x86_feature_detected!("avx2"),
                avx512f: is_x86_feature_detected!("avx512f"),
                bmi1: is_x86_feature_detected!("bmi1"),
                bmi2: is_x86_feature_detected!("bmi2"),
                lzcnt: is_x86_feature_detected!("lzcnt"),
            }
        }
        
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self {
                sse42: false,
                avx: false,
                avx2: false,
                avx512f: false,
                bmi1: false,
                bmi2: false,
                lzcnt: false,
            }
        }
    }

    /// Get the best available acceleration level
    pub fn best_level(&self) -&gt; SimdLevel {
        if self.avx512f {
            SimdLevel::Avx512
        } else if self.avx2 {
            SimdLevel::Avx2
        } else if self.avx {
            SimdLevel::Avx
        } else if self.sse42 {
            SimdLevel::Sse42
        } else {
            SimdLevel::Scalar
        }
    }

    /// Log detected capabilities
    pub fn log_capabilities(&self) {
        info!("SIMD Capabilities detected:");
        info!("  SSE4.2: {}", self.sse42);
        info!("  AVX: {}", self.avx);
        info!("  AVX2: {}", self.avx2);
        info!("  AVX-512F: {}", self.avx512f);
        info!("  BMI1: {}", self.bmi1);
        info!("  BMI2: {}", self.bmi2);
        info!("  LZCNT: {}", self.lzcnt);
        info!("  Best level: {:?}", self.best_level());
    }

    /// Check if hardware-accelerated LZCNT is available
    pub fn has_lzcnt(&self) -> bool {
        self.lzcnt || self.bmi1
    }
}

impl Default for SimdCapabilities {
    fn default() -> Self {
        Self::detect()
    }
}

/// SIMD acceleration level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdLevel {
    /// No SIMD, scalar operations only
    Scalar,
    /// SSE4.2 (2-wide double precision)
    Sse42,
    /// AVX (4-wide double precision)
    Avx,
    /// AVX2 (4-wide with FMA)
    Avx2,
    /// AVX-512 (8-wide double precision)
    Avx512,
}

impl SimdLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            SimdLevel::Scalar => "scalar",
            SimdLevel::Sse42 => "sse4.2",
            SimdLevel::Avx => "avx",
            SimdLevel::Avx2 => "avx2",
            SimdLevel::Avx512 => "avx512",
        }
    }
}

/// Unified SIMD accelerator that selects best available implementation
pub struct UnifiedSimdAccelerator {
    capabilities: SimdCapabilities,
    avx2_accelerator: Avx2Accelerator,
    sse_accelerator: SseAccelerator,
}

impl UnifiedSimdAccelerator {
    /// Create new unified accelerator
    pub fn new() -> Self {
        let capabilities = SimdCapabilities::detect();
        
        Self {
            capabilities,
            avx2_accelerator: Avx2Accelerator::new(),
            sse_accelerator: SseAccelerator::new(),
        }
    }

    /// Get detected capabilities
    pub fn capabilities(&self) -> &SimdCapabilities {
        &self.capabilities
    }

    /// Compute Z-scores using best available implementation
    pub fn compute_zscores(&self, values: &[f64], mean: f64, std_dev: f64) -> Vec<f64> {
        match self.capabilities.best_level() {
            SimdLevel::Avx512 | SimdLevel::Avx2 => {
                self.avx2_accelerator.compute_zscores(values, mean, std_dev)
            }
            SimdLevel::Avx | SimdLevel::Sse42 => {
                self.sse_accelerator.compute_zscores(values, mean, std_dev)
            }
            SimdLevel::Scalar => {
                values.iter().map(|&v| (v - mean) / std_dev).collect()
            }
        }
    }

    /// Find maximum using best available implementation
    pub fn find_max(&self, values: &[f64]) -> f64 {
        match self.capabilities.best_level() {
            SimdLevel::Avx512 | SimdLevel::Avx2 => {
                self.avx2_accelerator.find_max(values)
            }
            SimdLevel::Avx | SimdLevel::Sse42 => {
                self.sse_accelerator.find_max(values)
            }
            SimdLevel::Scalar => {
                values.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            }
        }
    }

    /// Find minimum using best available implementation
    pub fn find_min(&self, values: &[f64]) -> f64 {
        match self.capabilities.best_level() {
            SimdLevel::Avx512 | SimdLevel::Avx2 => {
                self.avx2_accelerator.find_min(values)
            }
            SimdLevel::Avx | SimdLevel::Sse42 => {
                self.sse_accelerator.find_min(values)
            }
            SimdLevel::Scalar => {
                values.iter().cloned().fold(f64::INFINITY, f64::min)
            }
        }
    }

    /// Compute microprice using dispatch to best implementation
    pub fn compute_microprice(
        &self,
        bids: &[f64],
        asks: &[f64],
        bid_sizes: &[f64],
        ask_sizes: &[f64],
    ) -> f64 {
        compute_microprice_dispatch(bids, asks, bid_sizes, ask_sizes)
    }
}

impl Default for UnifiedSimdAccelerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch processor for order book updates using SIMD
pub struct OrderBookSimdProcessor {
    accelerator: UnifiedSimdAccelerator,
}

impl OrderBookSimdProcessor {
    pub fn new() -> Self {
        Self {
            accelerator: UnifiedSimdAccelerator::new(),
        }
    }

    /// Update prices and compute new microprice
    pub fn update_and_compute_microprice(
        &self,
        bids: &mut [f64],
        asks: &mut [f64],
        bid_sizes: &[f64],
        ask_sizes: &[f64],
    ) -> f64 {
        self.accelerator.compute_microprice(bids, asks, bid_sizes, ask_sizes)
    }

    /// Normalize a price series using Z-score
    pub fn normalize_prices(&self, prices: &[f64]) -> Vec<f64> {
        if prices.is_empty() {
            return Vec::new();
        }

        // Compute mean
        let mean: f64 = prices.iter().sum::<f64>() / prices.len() as f64;

        // Compute standard deviation using SIMD
        let variance: f64 = prices
            .iter()
            .map(|&p| (p - mean).powi(2))
            .sum::<f64>() / prices.len() as f64;
        
        let std_dev = variance.sqrt().max(1e-10);

        // Compute Z-scores using SIMD
        self.accelerator.compute_zscores(prices, mean, std_dev)
    }

    /// Find price extremes
    pub fn find_extremes(&self, prices: &[f64]) -> (f64, f64) {
        let min = self.accelerator.find_min(prices);
        let max = self.accelerator.find_max(prices);
        (min, max)
    }
}

impl Default for OrderBookSimdProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capability_detection() {
        let caps = SimdCapabilities::detect();
        
        #[cfg(target_arch = "x86_64")]
        {
            println!("Running on x86_64");
            println!("Best SIMD level: {:?}", caps.best_level());
        }
        
        assert!(caps.best_level() >= SimdLevel::Scalar);
    }

    #[test]
    fn test_unified_accelerator() {
        let accelerator = UnifiedSimdAccelerator::new();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        let zscores = accelerator.compute_zscores(&values, 3.0, 1.414);
        assert_eq!(zscores.len(), 5);
        
        let max = accelerator.find_max(&values);
        assert_eq!(max, 5.0);
        
        let min = accelerator.find_min(&values);
        assert_eq!(min, 1.0);
    }

    #[test]
    fn test_orderbook_processor() {
        let processor = OrderBookSimdProcessor::new();
        let prices = vec![100.0, 101.0, 99.0, 102.0, 98.0];
        
        let normalized = processor.normalize_prices(&prices);
        assert_eq!(normalized.len(), 5);
        
        let (min, max) = processor.find_extremes(&prices);
        assert_eq!(min, 98.0);
        assert_eq!(max, 102.0);
    }

    #[test]
    fn test_simd_level_ordering() {
        assert!(SimdLevel::Avx512 > SimdLevel::Avx2);
        assert!(SimdLevel::Avx2 > SimdLevel::Avx);
        assert!(SimdLevel::Avx > SimdLevel::Sse42);
        assert!(SimdLevel::Sse42 > SimdLevel::Scalar);
    }
}
