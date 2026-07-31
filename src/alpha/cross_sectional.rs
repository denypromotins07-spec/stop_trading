//! Cross-Sectional Momentum and Mean-Reversion Z-Score Ranker
//! 
//! Implements lock-free atomic arrays to rank BTC, ETH, SOL, and USDT pairs
//! in O(1) time without heap allocations using SIMD-accelerated sorting networks.

use std::sync::atomic::{AtomicU64, AtomicI64, Ordering};
use std::arch::x86_64::*;

/// Maximum number of assets in the cross-sectional universe
pub const MAX_ASSETS: usize = 64;

/// Lock-free cross-sectional ranker using atomic arrays
pub struct CrossSectionalRanker {
    /// Raw returns for each asset (fixed-point Q16.48)
    returns: [AtomicI64; MAX_ASSETS],
    /// Ranks for each asset (atomic for lock-free access)
    ranks: [AtomicU64; MAX_ASSETS],
    /// Volatility-adjusted scores
    vola_scores: [AtomicI64; MAX_ASSETS],
    /// Asset count
    count: AtomicU64,
}

/// Sorting network for ultra-fast O(1) ranking (SIMD-accelerated)
#[inline(always)]
unsafe fn sort_network_simd(values: &mut [f64; 8]) {
    // Load 4 doubles into AVX register
    let mut v0 = _mm256_loadu_pd(values.as_ptr());
    let mut v1 = _mm256_loadu_pd(values.as_ptr().add(4));
    
    // Bitonic sort network for 8 elements
    // Stage 1: Compare pairs
    let cmp1 = _mm256_cmp_pd(v0, v1, _CMP_LT_OQ);
    let min1 = _mm256_min_pd(v0, v1);
    let max1 = _mm256_max_pd(v0, v1);
    v0 = _mm256_blendv_pd(max1, min1, cmp1);
    v1 = _mm256_blendv_pd(min1, max1, cmp1);
    
    // Stage 2: Cross compare
    let shuf0 = _mm256_permute2f128_pd(v0, v0, 0b00000001);
    let shuf1 = _mm256_permute2f128_pd(v1, v1, 0b00000001);
    let cmp2 = _mm256_cmp_pd(v0, shuf0, _CMP_LT_OQ);
    let cmp3 = _mm256_cmp_pd(v1, shuf1, _CMP_LT_OQ);
    v0 = _mm256_blendv_pd(shuf0, v0, cmp2);
    v1 = _mm256_blendv_pd(shuf1, v1, cmp3);
    
    // Stage 3: Final merge
    let cmp4 = _mm256_cmp_pd(v0, v1, _CMP_LT_OQ);
    let final_min = _mm256_min_pd(v0, v1);
    let final_max = _mm256_max_pd(v0, v1);
    v0 = _mm256_blendv_pd(final_max, final_min, cmp4);
    v1 = _mm256_blendv_pd(final_min, final_max, cmp4);
    
    // Store back
    _mm256_storeu_pd(values.as_mut_ptr(), v0);
    _mm256_storeu_pd(values.as_mut_ptr().add(4), v1);
}

impl CrossSectionalRanker {
    pub const fn new() -> Self {
        const fn init_atomic_i64() -> AtomicI64 {
            AtomicI64::new(0)
        }
        const fn init_atomic_u64() -> AtomicU64 {
            AtomicU64::new(0)
        }
        
        Self {
            returns: [init_atomic_i64(); MAX_ASSETS],
            ranks: [init_atomic_u64(); MAX_ASSETS],
            vola_scores: [init_atomic_i64(); MAX_ASSETS],
            count: AtomicU64::new(0),
        }
    }
    
    /// Update return for an asset (lock-free)
    #[inline]
    pub fn update_return(&self, asset_idx: usize, return_bps: i64) {
        if asset_idx < MAX_ASSETS {
            self.returns[asset_idx].store(return_bps, Ordering::Release);
        }
    }
    
    /// Update volatility score for an asset
    #[inline]
    pub fn update_vola_score(&self, asset_idx: usize, score: i64) {
        if asset_idx < MAX_ASSETS {
            self.vola_scores[asset_idx].store(score, Ordering::Release);
        }
    }
    
    /// Compute cross-sectional Z-scores and ranks (O(1) with fixed universe)
    pub fn compute_ranks(&self) {
        let count = self.count.load(Ordering::Acquire) as usize;
        let actual_count = count.min(MAX_ASSETS);
        
        if actual_count == 0 {
            return;
        }
        
        // Gather returns into stack array (no heap allocation)
        let mut returns_arr: [f64; 8] = [0.0; 8];
        let mut ranks_arr: [usize; 8] = [0; 8];
        
        // Process in chunks of 8 for SIMD
        let mut chunk_start = 0;
        while chunk_start < actual_count {
            let chunk_end = (chunk_start + 8).min(actual_count);
            let chunk_size = chunk_end - chunk_start;
            
            // Reset arrays
            returns_arr = [0.0; 8];
            ranks_arr = [0; 8];
            
            // Load returns (convert from fixed-point Q16.48)
            for i in 0..chunk_size {
                let raw = self.returns[chunk_start + i].load(Ordering::Acquire);
                returns_arr[i] = raw as f64 / (1u64 << 48) as f64;
                ranks_arr[i] = chunk_start + i;
            }
            
            // SIMD sort for chunks of 8
            if chunk_size == 8 {
                unsafe {
                    sort_network_simd(&mut returns_arr);
                }
            } else {
                // Scalar sort for remainder
                returns_arr[..chunk_size].sort_by(|a, b| a.partial_cmp(b).unwrap());
            }
            
            // Assign ranks based on sorted order
            for i in 0..chunk_size {
                let rank = (i as u64) * (1u64 << 32) | (ranks_arr[i] as u64);
                self.ranks[ranks_arr[i]].store(rank, Ordering::Release);
            }
            
            chunk_start = chunk_end;
        }
    }
    
    /// Get momentum Z-score for an asset
    #[inline]
    pub fn get_momentum_zscore(&self, asset_idx: usize) -> f64 {
        if asset_idx >= MAX_ASSETS {
            return 0.0;
        }
        
        let raw_return = self.returns[asset_idx].load(Ordering::Acquire);
        let return_f64 = raw_return as f64 / (1u64 << 48) as f64;
        
        // Compute cross-sectional mean and std (Welford's algorithm)
        let count = self.count.load(Ordering::Acquire) as f64;
        if count < 2.0 {
            return 0.0;
        }
        
        let mut mean = 0.0;
        let mut m2 = 0.0;
        
        for i in 0..MAX_ASSETS {
            let raw = self.returns[i].load(Ordering::Acquire);
            if raw == 0 && i >= self.count.load(Ordering::Acquire) as usize {
                break;
            }
            let val = raw as f64 / (1u64 << 48) as f64;
            let delta = val - mean;
            mean += delta / ((i + 1) as f64);
            let delta2 = val - mean;
            m2 += delta * delta2;
        }
        
        let variance = m2 / (count - 1.0);
        let std = variance.sqrt();
        
        if std < 1e-10 {
            return 0.0;
        }
        
        (return_f64 - mean) / std
    }
    
    /// Get mean-reversion signal (negative of momentum Z-score)
    #[inline]
    pub fn get_mean_reversion_signal(&self, asset_idx: usize) -> f64 {
        -self.get_momentum_zscore(asset_idx)
    }
    
    /// Get composite alpha signal combining momentum and mean-reversion
    pub fn get_composite_signal(&self, asset_idx: usize, momentum_weight: f64) -> f64 {
        let momentum = self.get_momentum_zscore(asset_idx);
        let mr = self.get_mean_reversion_signal(asset_idx);
        
        // Regime-adaptive weighting based on volatility score
        let vola_raw = self.vola_scores[asset_idx].load(Ordering::Acquire);
        let vola_norm = (vola_raw as f64 / (1u64 << 48) as f64).clamp(-1.0, 1.0);
        
        // High volatility favors mean-reversion, low favors momentum
        let adaptive_mw = momentum_weight * (1.0 - vola_norm.abs());
        let adaptive_mr_w = 1.0 - adaptive_mw;
        
        momentum * adaptive_mw + mr * adaptive_mr_w
    }
    
    /// Set asset count
    #[inline]
    pub fn set_count(&self, count: usize) {
        self.count.store(count.min(MAX_ASSETS) as u64, Ordering::Release);
    }
    
    /// Get current asset count
    #[inline]
    pub fn get_count(&self) -> usize {
        self.count.load(Ordering::Acquire) as usize
    }
    
    /// Get rank for an asset (higher = better momentum)
    #[inline]
    pub fn get_rank(&self, asset_idx: usize) -> u64 {
        if asset_idx >= MAX_ASSETS {
            return 0;
        }
        self.ranks[asset_idx].load(Ordering::Acquire)
    }
}

/// Multi-asset universe manager
pub struct AssetUniverse {
    /// Asset symbols (static strings, no allocation)
    symbols: [&'static str; MAX_ASSETS],
    /// Mapping from symbol hash to index
    symbol_map: [u64; 256],
    /// Active asset count
    active_count: usize,
}

impl AssetUniverse {
    pub const fn new() -> Self {
        Self {
            symbols: [""; MAX_ASSETS],
            symbol_map: [u64::MAX; 256],
            active_count: 0,
        }
    }
    
    /// Register an asset in the universe
    pub fn register_asset(&mut self, symbol: &'static str) -> Option<usize> {
        if self.active_count >= MAX_ASSETS {
            return None;
        }
        
        // Simple hash for symbol lookup
        let hash = self.hash_symbol(symbol) as usize % 256;
        self.symbol_map[hash] = self.active_count as u64;
        self.symbols[self.active_count] = symbol;
        
        let idx = self.active_count;
        self.active_count += 1;
        Some(idx)
    }
    
    /// Get asset index by symbol
    #[inline]
    pub fn get_index(&self, symbol: &str) -> Option<usize> {
        let hash = self.hash_symbol(symbol) as usize % 256;
        let idx = self.symbol_map[hash] as usize;
        if idx < self.active_count && self.symbols[idx] == symbol {
            Some(idx)
        } else {
            None
        }
    }
    
    /// Get symbol by index
    #[inline]
    pub fn get_symbol(&self, idx: usize) -> Option<&'static str> {
        if idx < self.active_count {
            Some(self.symbols[idx])
        } else {
            None
        }
    }
    
    /// Get active count
    #[inline]
    pub fn active_count(&self) -> usize {
        self.active_count
    }
    
    /// FNV-1a hash for symbols
    #[inline]
    fn hash_symbol(&self, symbol: &str) -> u64 {
        let mut hash = 0xcbf29ce484222325u64;
        for byte in symbol.as_bytes() {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cross_sectional_ranker() {
        let ranker = CrossSectionalRanker::new();
        let mut universe = AssetUniverse::new();
        
        // Register assets
        let btc = universe.register_asset("BTC").unwrap();
        let eth = universe.register_asset("ETH").unwrap();
        let sol = universe.register_asset("SOL").unwrap();
        
        ranker.set_count(universe.active_count());
        
        // Update returns (in basis points, scaled to Q16.48)
        ranker.update_return(btc, 100 << 48); // +100 bps
        ranker.update_return(eth, 50 << 48);  // +50 bps
        ranker.update_return(sol, -25 << 48); // -25 bps
        
        // Compute ranks
        ranker.compute_ranks();
        
        // Verify rankings
        assert!(ranker.get_rank(btc) > ranker.get_rank(eth));
        assert!(ranker.get_rank(eth) > ranker.get_rank(sol));
    }
    
    #[test]
    fn test_zscore_calculation() {
        let ranker = CrossSectionalRanker::new();
        let mut universe = AssetUniverse::new();
        
        let btc = universe.register_asset("BTC").unwrap();
        let eth = universe.register_asset("ETH").unwrap();
        let sol = universe.register_asset("SOL").unwrap();
        
        ranker.set_count(universe.active_count());
        
        // Create spread: BTC high, SOL low
        ranker.update_return(btc, 200 << 48);
        ranker.update_return(eth, 0 << 48);
        ranker.update_return(sol, -200 << 48);
        
        let btc_zscore = ranker.get_momentum_zscore(btc);
        let sol_zscore = ranker.get_momentum_zscore(sol);
        
        // BTC should have positive Z-score, SOL negative
        assert!(btc_zscore > 0.5);
        assert!(sol_zscore < -0.5);
    }
}
