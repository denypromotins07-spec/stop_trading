//! Multi-Factor Alpha Blending Engine
//! 
//! Combines orthogonal signals (Value, Momentum, Quality, Volatility) into a single
//! composite score using dynamic, lock-free weights for concurrent per-symbol actors.

use std::sync::atomic::{AtomicU64, AtomicI64, Ordering};
use std::arch::x86_64::*;

/// Maximum number of factors in the model
pub const MAX_FACTORS: usize = 16;

/// Maximum number of assets
pub const MAX_ASSETS: usize = 64;

/// Factor types supported by the engine
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FactorType {
    Value = 0,
    Momentum = 1,
    Quality = 2,
    Volatility = 3,
    Liquidity = 4,
    Sentiment = 5,
    Custom = 6,
}

/// Lock-free multi-factor alpha blending engine
pub struct MultiFactorEngine {
    /// Factor scores per asset (Q16.48 fixed-point)
    factor_scores: [[AtomicI64; MAX_ASSETS]; MAX_FACTORS],
    /// Factor weights (Q16.48 fixed-point, sum to 1.0)
    factor_weights: [AtomicI64; MAX_FACTORS],
    /// Composite alpha scores per asset
    composite_scores: [AtomicI64; MAX_ASSETS],
    /// Asset active flags
    asset_active: [AtomicU64; MAX_ASSETS],
    /// Number of active factors
    num_factors: AtomicU64,
    /// Number of active assets
    num_assets: AtomicU64,
    /// Orthogonality correction matrix (simplified diagonal)
    ortho_correction: [AtomicI64; MAX_FACTORS],
}

/// SIMD-accelerated dot product for factor blending
#[inline(always)]
unsafe fn simd_dot_product(weights: &[f64; 8], scores: &[f64; 8]) -> f64 {
    let w = _mm256_loadu_pd(weights.as_ptr());
    let w_hi = _mm256_loadu_pd(weights.as_ptr().add(4));
    let s = _mm256_loadu_pd(scores.as_ptr());
    let s_hi = _mm256_loadu_pd(scores.as_ptr().add(4));
    
    let prod_lo = _mm256_mul_pd(w, s);
    let prod_hi = _mm256_mul_pd(w_hi, s_hi);
    
    // Horizontal add
    let hadd1 = _mm256_hadd_pd(prod_lo, prod_hi);
    let hadd2 = _mm256_permute2f128_pd(hadd1, hadd1, 0b00000001);
    let final_sum = _mm256_add_pd(hadd1, hadd2);
    
    let mut result = [0.0f64; 4];
    _mm256_storeu_pd(result.as_mut_ptr(), final_sum);
    
    result[0] + result[1]
}

impl MultiFactorEngine {
    pub const fn new() -> Self {
        const fn init_atomic_i64() -> AtomicI64 {
            AtomicI64::new(0)
        }
        const fn init_atomic_u64() -> AtomicU64 {
            AtomicU64::new(0)
        }
        
        Self {
            factor_scores: [[init_atomic_i64(); MAX_ASSETS]; MAX_FACTORS],
            factor_weights: [init_atomic_i64(); MAX_FACTORS],
            composite_scores: [init_atomic_i64(); MAX_ASSETS],
            asset_active: [init_atomic_u64(); MAX_ASSETS],
            num_factors: AtomicU64::new(0),
            num_assets: AtomicU64::new(0),
            ortho_correction: [init_atomic_i64(); MAX_FACTORS],
        }
    }
    
    /// Register a factor with initial weight
    pub fn register_factor(&self, factor_type: FactorType, initial_weight: f64) -> Option<usize> {
        let factor_idx = factor_type as usize;
        if factor_idx >= MAX_FACTORS {
            return None;
        }
        
        // Convert weight to Q16.48
        let weight_fixed = (initial_weight * (1u64 << 48) as f64) as i64;
        self.factor_weights[factor_idx].store(weight_fixed, Ordering::Release);
        
        // Default orthogonality correction (identity)
        self.ortho_correction[factor_idx].store((1u64 << 48) as i64, Ordering::Release);
        
        // Update factor count if new
        let current = self.num_factors.load(Ordering::Acquire);
        if factor_idx >= current as usize {
            self.num_factors.store((factor_idx + 1) as u64, Ordering::Release);
        }
        
        Some(factor_idx)
    }
    
    /// Update factor weight dynamically (lock-free)
    #[inline]
    pub fn update_factor_weight(&self, factor_type: FactorType, new_weight: f64) {
        let factor_idx = factor_type as usize;
        if factor_idx < MAX_FACTORS {
            let weight_fixed = (new_weight * (1u64 << 48) as f64) as i64;
            self.factor_weights[factor_idx].store(weight_fixed, Ordering::Release);
        }
    }
    
    /// Update factor score for an asset
    #[inline]
    pub fn update_factor_score(&self, factor_type: FactorType, asset_idx: usize, score: f64) {
        let factor_idx = factor_type as usize;
        if factor_idx < MAX_FACTORS && asset_idx < MAX_ASSETS {
            let score_fixed = (score * (1u64 << 48) as f64) as i64;
            self.factor_scores[factor_idx][asset_idx].store(score_fixed, Ordering::Release);
            self.asset_active[asset_idx].store(1, Ordering::Release);
        }
    }
    
    /// Set orthogonality correction for a factor (decorrelation coefficient)
    #[inline]
    pub fn set_orthogonality_correction(&self, factor_type: FactorType, correction: f64) {
        let factor_idx = factor_type as usize;
        if factor_idx < MAX_FACTORS {
            let corr_fixed = (correction * (1u64 << 48) as f64) as i64;
            self.ortho_correction[factor_idx].store(corr_fixed, Ordering::Release);
        }
    }
    
    /// Compute composite alpha score for all assets (SIMD-accelerated)
    pub fn compute_composite_scores(&self) {
        let num_factors = self.num_factors.load(Ordering::Acquire) as usize;
        let num_assets = self.num_assets.load(Ordering::Acquire) as usize;
        
        if num_factors == 0 || num_assets == 0 {
            return;
        }
        
        // Process each asset
        for asset_idx in 0..num_assets.min(MAX_ASSETS) {
            if self.asset_active[asset_idx].load(Ordering::Acquire) == 0 {
                continue;
            }
            
            // Gather factor scores and weights
            let mut scores_arr: [f64; 8] = [0.0; 8];
            let mut weights_arr: [f64; 8] = [0.0; 8];
            
            let mut composite = 0.0f64;
            let mut processed = 0;
            
            while processed < num_factors {
                let chunk_size = (num_factors - processed).min(8);
                
                // Reset arrays
                scores_arr = [0.0; 8];
                weights_arr = [0.0; 8];
                
                // Load scores and weights
                for i in 0..chunk_size {
                    let factor_idx = processed + i;
                    let score_raw = self.factor_scores[factor_idx][asset_idx]
                        .load(Ordering::Acquire);
                    let weight_raw = self.factor_weights[factor_idx]
                        .load(Ordering::Acquire);
                    let ortho_raw = self.ortho_correction[factor_idx]
                        .load(Ordering::Acquire);
                    
                    let score_f64 = score_raw as f64 / (1u64 << 48) as f64;
                    let weight_f64 = weight_raw as f64 / (1u64 << 48) as f64;
                    let ortho_f64 = ortho_raw as f64 / (1u64 << 48) as f64;
                    
                    // Apply orthogonality correction
                    scores_arr[i] = score_f64 * ortho_f64;
                    weights_arr[i] = weight_f64;
                }
                
                // SIMD dot product for full chunks
                if chunk_size == 8 {
                    unsafe {
                        composite += simd_dot_product(&weights_arr, &scores_arr);
                    }
                } else {
                    // Scalar for remainder
                    for i in 0..chunk_size {
                        composite += weights_arr[i] * scores_arr[i];
                    }
                }
                
                processed += chunk_size;
            }
            
            // Store composite score
            let composite_fixed = (composite * (1u64 << 48) as f64) as i64;
            self.composite_scores[asset_idx].store(composite_fixed, Ordering::Release);
        }
    }
    
    /// Get composite score for an asset
    #[inline]
    pub fn get_composite_score(&self, asset_idx: usize) -> f64 {
        if asset_idx >= MAX_ASSETS {
            return 0.0;
        }
        let raw = self.composite_scores[asset_idx].load(Ordering::Acquire);
        raw as f64 / (1u64 << 48) as f64
    }
    
    /// Get ranked assets by composite score (returns indices sorted by score)
    pub fn get_ranked_assets(&self, max_results: usize) -> Vec<(usize, f64)> {
        let num_assets = self.num_assets.load(Ordering::Acquire) as usize;
        let mut results: [(usize, f64); MAX_ASSETS] = [(0, 0.0); MAX_ASSETS];
        let mut count = 0;
        
        for idx in 0..num_assets.min(MAX_ASSETS) {
            if self.asset_active[idx].load(Ordering::Acquire) != 0 {
                let score = self.get_composite_score(idx);
                results[count] = (idx, score);
                count += 1;
            }
        }
        
        // Sort by score descending (simple insertion sort for small arrays)
        for i in 1..count {
            let key = results[i];
            let mut j = i;
            while j > 0 && results[j - 1].1 < key.1 {
                results[j] = results[j - 1];
                j -= 1;
            }
            results[j] = key;
        }
        
        results[..count.min(max_results)].to_vec()
    }
    
    /// Normalize weights to sum to 1.0
    pub fn normalize_weights(&self) {
        let num_factors = self.num_factors.load(Ordering::Acquire) as usize;
        
        let mut total: f64 = 0.0;
        for i in 0..num_factors.min(MAX_FACTORS) {
            let raw = self.factor_weights[i].load(Ordering::Acquire);
            total += raw as f64 / (1u64 << 48) as f64;
        }
        
        if total.abs() < 1e-10 {
            // Equal weight fallback
            let equal_weight = 1.0 / num_factors.max(1) as f64;
            for i in 0..num_factors.min(MAX_FACTORS) {
                let w_fixed = (equal_weight * (1u64 << 48) as f64) as i64;
                self.factor_weights[i].store(w_fixed, Ordering::Release);
            }
            return;
        }
        
        // Normalize
        for i in 0..num_factors.min(MAX_FACTORS) {
            let raw = self.factor_weights[i].load(Ordering::Acquire);
            let normalized = (raw as f64 / (1u64 << 48) as f64) / total;
            let w_fixed = (normalized * (1u64 << 48) as f64) as i64;
            self.factor_weights[i].store(w_fixed, Ordering::Release);
        }
    }
    
    /// Set number of active assets
    #[inline]
    pub fn set_num_assets(&self, count: usize) {
        self.num_assets.store(count.min(MAX_ASSETS) as u64, Ordering::Release);
    }
    
    /// Get number of active assets
    #[inline]
    pub fn get_num_assets(&self) -> usize {
        self.num_assets.load(Ordering::Acquire) as usize
    }
    
    /// Deactivate an asset
    #[inline]
    pub fn deactivate_asset(&self, asset_idx: usize) {
        if asset_idx < MAX_ASSETS {
            self.asset_active[asset_idx].store(0, Ordering::Release);
        }
    }
}

/// Factor signal quality metrics
#[derive(Debug, Clone)]
pub struct FactorMetrics {
    pub ic: f64,           // Information Coefficient
    pub ic_decay: f64,     // IC decay rate
    pub turnover: f64,     // Factor turnover
    pub sharpe: f64,       // Factor Sharpe ratio
    pub max_drawdown: f64, // Max drawdown of factor portfolio
}

impl FactorMetrics {
    pub const fn new() -> Self {
        Self {
            ic: 0.0,
            ic_decay: 0.0,
            turnover: 0.0,
            sharpe: 0.0,
            max_drawdown: 0.0,
        }
    }
}

/// Real-time factor performance tracker
pub struct FactorTracker {
    /// Rolling IC storage (ring buffer)
    ic_history: [f64; 256],
    /// Head pointer
    head: usize,
    /// Count of valid entries
    count: usize,
    /// Metrics per factor
    metrics: [FactorMetrics; MAX_FACTORS],
}

impl FactorTracker {
    pub const fn new() -> Self {
        Self {
            ic_history: [0.0; 256],
            head: 0,
            count: 0,
            metrics: [FactorMetrics::new(); MAX_FACTORS],
        }
    }
    
    /// Record IC observation
    pub fn record_ic(&mut self, ic: f64) {
        self.ic_history[self.head] = ic;
        self.head = (self.head + 1) % 256;
        if self.count < 256 {
            self.count += 1;
        }
    }
    
    /// Compute rolling mean IC
    pub fn rolling_mean_ic(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let mut sum = 0.0;
        for i in 0..self.count {
            sum += self.ic_history[i];
        }
        sum / self.count as f64
    }
    
    /// Compute IC standard deviation
    pub fn rolling_std_ic(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        let mean = self.rolling_mean_ic();
        let mut var = 0.0;
        for i in 0..self.count {
            let diff = self.ic_history[i] - mean;
            var += diff * diff;
        }
        (var / (self.count - 1) as f64).sqrt()
    }
    
    /// Get IR (Information Ratio)
    pub fn information_ratio(&self) -> f64 {
        let std = self.rolling_std_ic();
        if std < 1e-10 {
            return 0.0;
        }
        self.rolling_mean_ic() / std
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_multi_factor_engine() {
        let engine = MultiFactorEngine::new();
        
        // Register factors
        engine.register_factor(FactorType::Momentum, 0.4);
        engine.register_factor(FactorType::Value, 0.3);
        engine.register_factor(FactorType::Quality, 0.2);
        engine.register_factor(FactorType::Volatility, 0.1);
        
        engine.normalize_weights();
        engine.set_num_assets(3);
        
        // Update scores for 3 assets
        engine.update_factor_score(FactorType::Momentum, 0, 1.0);
        engine.update_factor_score(FactorType::Value, 0, 0.5);
        engine.update_factor_score(FactorType::Quality, 0, 0.8);
        engine.update_factor_score(FactorType::Volatility, 0, -0.2);
        
        engine.update_factor_score(FactorType::Momentum, 1, 0.2);
        engine.update_factor_score(FactorType::Value, 1, 1.0);
        engine.update_factor_score(FactorType::Quality, 1, 0.6);
        engine.update_factor_score(FactorType::Volatility, 1, 0.1);
        
        engine.update_factor_score(FactorType::Momentum, 2, -0.5);
        engine.update_factor_score(FactorType::Value, 2, -0.3);
        engine.update_factor_score(FactorType::Quality, 2, -0.8);
        engine.update_factor_score(FactorType::Volatility, 2, 0.9);
        
        // Compute composites
        engine.compute_composite_scores();
        
        // Verify ranking
        let ranked = engine.get_ranked_assets(3);
        assert_eq!(ranked.len(), 3);
        assert!(ranked[0].1 > ranked[1].1);
        assert!(ranked[1].1 > ranked[2].1);
    }
    
    #[test]
    fn test_factor_tracker() {
        let mut tracker = FactorTracker::new();
        
        // Record some IC values
        for i in 0..50 {
            tracker.record_ic(0.05 + (i as f64 * 0.001));
        }
        
        let mean_ic = tracker.rolling_mean_ic();
        let ir = tracker.information_ratio();
        
        assert!(mean_ic > 0.05);
        assert!(ir > 0.0);
    }
}
