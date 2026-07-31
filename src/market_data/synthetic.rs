//! Synthetic Index Constructor
//! 
//! Builds synthetic indices (e.g., Crypto10, DeFi Index) using real-time market cap and volume weighting.
//! Generates proprietary benchmark tick streams for relative-value alpha evaluation.

use std::sync::atomic::{AtomicU64, AtomicI64, Ordering};

/// Maximum components in a synthetic index
pub const MAX_COMPONENTS: usize = 32;

/// Weighting methodology
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WeightingMethod {
    MarketCap = 0,
    Volume = 1,
    Equal = 2,
    Fundamental = 3,
    Custom = 4,
}

/// Component of a synthetic index
#[derive(Debug, Clone)]
pub struct IndexComponent {
    pub symbol: &'static str,
    pub weight: f64,          // Target weight
    pub current_weight: f64,  // Real-time weight
    pub price: i64,           // Q16.48 fixed-point
    pub market_cap: u64,      // Market cap in USD
    pub volume_24h: u64,      // 24h volume in USD
}

/// Synthetic index tick
#[derive(Debug, Clone)]
pub struct IndexTick {
    pub index_value: i64,     // Q16.48 fixed-point
    pub change_bps: f64,      // Change in basis points
    pub timestamp_ns: u64,
    pub volume: i64,          // Aggregated volume (Q16.48)
}

/// Synthetic index engine
pub struct SyntheticIndex {
    /// Index name
    pub name: &'static str,
    /// Components
    components: [Option<IndexComponent>; MAX_COMPONENTS],
    /// Number of active components
    component_count: AtomicU64,
    /// Base value (e.g., 1000.0)
    base_value: f64,
    /// Current index value (Q16.48)
    current_value: AtomicI64,
    /// Previous value for change calculation
    previous_value: AtomicI64,
    /// Weighting method
    weighting_method: WeightingMethod,
    /// Rebalance threshold (weight drift %)
    rebalance_threshold: f64,
    /// Last rebalance timestamp
    last_rebalance_ns: AtomicU64,
}

impl SyntheticIndex {
    pub const fn new(name: &'static str, base_value: f64) -> Self {
        Self {
            name,
            components: [None; MAX_COMPONENTS],
            component_count: AtomicU64::new(0),
            base_value,
            current_value: AtomicI64::new(0),
            previous_value: AtomicI64::new(0),
            weighting_method: WeightingMethod::MarketCap,
            rebalance_threshold: 0.05, // 5% drift triggers rebalance
            last_rebalance_ns: AtomicU64::new(0),
        }
    }
    
    /// Add a component to the index
    pub fn add_component(&mut self, symbol: &'static str, initial_weight: f64) -> bool {
        let count = self.component_count.load(Ordering::Acquire);
        if count >= MAX_COMPONENTS as u64 {
            return false;
        }
        
        self.components[count as usize] = Some(IndexComponent {
            symbol,
            weight: initial_weight,
            current_weight: initial_weight,
            price: 0,
            market_cap: 0,
            volume_24h: 0,
        });
        
        self.component_count.store(count + 1, Ordering::Release);
        true
    }
    
    /// Set weighting method
    #[inline]
    pub fn set_weighting_method(&mut self, method: WeightingMethod) {
        self.weighting_method = method;
    }
    
    /// Update component price and recalculate index
    pub fn update_price(&self, symbol: &str, price: i64) -> Option<IndexTick> {
        let count = self.component_count.load(Ordering::Acquire);
        let mut found_idx = None;
        
        // Find component
        for i in 0..count as usize {
            if let Some(ref comp) = self.components[i] {
                if comp.symbol == symbol {
                    found_idx = Some(i);
                    break;
                }
            }
        }
        
        let idx = found_idx?;
        
        // Update price
        if let Some(ref mut comp) = &mut self.components[idx] {
            let old_price = comp.price;
            comp.price = price;
            
            // Recalculate weights based on method
            self.recalculate_weights();
        }
        
        // Calculate new index value
        Some(self.calculate_index_tick())
    }
    
    /// Recalculate component weights based on methodology
    fn recalculate_weights(&self) {
        let count = self.component_count.load(Ordering::Acquire);
        
        match self.weighting_method {
            WeightingMethod::MarketCap => {
                let total_cap: u64 = self.components[..count as usize]
                    .iter()
                    .filter_map(|c| c.as_ref())
                    .map(|c| c.market_cap)
                    .sum();
                
                if total_cap > 0 {
                    for i in 0..count as usize {
                        if let Some(ref mut comp) = &mut self.components[i] {
                            comp.current_weight = comp.market_cap as f64 / total_cap as f64;
                        }
                    }
                }
            }
            WeightingMethod::Volume => {
                let total_vol: u64 = self.components[..count as usize]
                    .iter()
                    .filter_map(|c| c.as_ref())
                    .map(|c| c.volume_24h)
                    .sum();
                
                if total_vol > 0 {
                    for i in 0..count as usize {
                        if let Some(ref mut comp) = &mut self.components[i] {
                            comp.current_weight = comp.volume_24h as f64 / total_vol as f64;
                        }
                    }
                }
            }
            WeightingMethod::Equal => {
                let equal_weight = 1.0 / count.max(1) as f64;
                for i in 0..count as usize {
                    if let Some(ref mut comp) = &mut self.components[i] {
                        comp.current_weight = equal_weight;
                    }
                }
            }
            _ => {}
        }
    }
    
    /// Calculate current index tick
    fn calculate_index_tick(&self) -> IndexTick {
        let count = self.component_count.load(Ordering::Acquire);
        let mut weighted_sum = 0.0f64;
        let mut total_volume: i64 = 0;
        
        for i in 0..count as usize {
            if let Some(ref comp) = self.components[i] {
                let price_f64 = comp.price as f64 / (1u64 << 48) as f64;
                weighted_sum += price_f64 * comp.current_weight;
                total_volume += comp.volume_24h as i64;
            }
        }
        
        // Scale to base value
        let index_value = (weighted_sum * self.base_value * (1u64 << 48) as f64) as i64;
        
        let prev_value = self.previous_value.load(Ordering::Acquire);
        let prev_f64 = prev_value as f64 / (1u64 << 48) as f64;
        let curr_f64 = index_value as f64 / (1u64 << 48) as f64;
        
        let change_bps = if prev_f64 > 0.0 {
            ((curr_f64 - prev_f64) / prev_f64) * 10000.0
        } else {
            0.0
        };
        
        // Update stored values
        self.previous_value.store(self.current_value.load(Ordering::Acquire), Ordering::Release);
        self.current_value.store(index_value, Ordering::Release);
        
        IndexTick {
            index_value,
            change_bps,
            timestamp_ns: get_timestamp_ns(),
            volume: total_volume,
        }
    }
    
    /// Check if rebalance is needed
    pub fn check_rebalance_needed(&self) -> bool {
        let count = self.component_count.load(Ordering::Acquire);
        
        for i in 0..count as usize {
            if let Some(ref comp) = self.components[i] {
                let drift = (comp.current_weight - comp.weight).abs();
                if drift > self.rebalance_threshold {
                    return true;
                }
            }
        }
        false
    }
    
    /// Get current index value
    #[inline]
    pub fn get_value(&self) -> f64 {
        self.current_value.load(Ordering::Acquire) as f64 / (1u64 << 48) as f64
    }
    
    /// Get component count
    #[inline]
    pub fn component_count(&self) -> usize {
        self.component_count.load(Ordering::Acquire) as usize
    }
    
    /// Set rebalance threshold
    #[inline]
    pub fn set_rebalance_threshold(&mut self, threshold: f64) {
        self.rebalance_threshold = threshold.clamp(0.01, 0.5);
    }
    
    /// Update component market cap
    pub fn update_market_cap(&self, symbol: &str, market_cap: u64) {
        let count = self.component_count.load(Ordering::Acquire);
        for i in 0..count as usize {
            if let Some(ref mut comp) = &mut self.components[i] {
                if comp.symbol == symbol {
                    comp.market_cap = market_cap;
                    if self.weighting_method == WeightingMethod::MarketCap {
                        self.recalculate_weights();
                    }
                    break;
                }
            }
        }
    }
    
    /// Update component volume
    pub fn update_volume(&self, symbol: &str, volume: u64) {
        let count = self.component_count.load(Ordering::Acquire);
        for i in 0..count as usize {
            if let Some(ref mut comp) = &mut self.components[i] {
                if comp.symbol == symbol {
                    comp.volume_24h = volume;
                    if self.weighting_method == WeightingMethod::Volume {
                        self.recalculate_weights();
                    }
                    break;
                }
            }
        }
    }
}

/// Get current timestamp in nanoseconds
#[inline]
fn get_timestamp_ns() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_synthetic_index() {
        let mut index = SyntheticIndex::new("Crypto10", 1000.0);
        
        // Add components
        index.add_component("BTC", 0.4);
        index.add_component("ETH", 0.3);
        index.add_component("SOL", 0.3);
        
        // Update prices
        let tick1 = index.update_price("BTC", 50000 << 48).unwrap();
        let tick2 = index.update_price("ETH", 3000 << 48).unwrap();
        let tick3 = index.update_price("SOL", 100 << 48).unwrap();
        
        assert!(tick3.index_value > 0);
        assert_eq!(index.component_count(), 3);
    }
    
    #[test]
    fn test_equal_weight() {
        let mut index = SyntheticIndex::new("EqualWeight3", 1000.0);
        index.set_weighting_method(WeightingMethod::Equal);
        
        index.add_component("A", 0.33);
        index.add_component("B", 0.33);
        index.add_component("C", 0.34);
        
        index.update_price("A", 100 << 48);
        index.update_price("B", 200 << 48);
        index.update_price("C", 300 << 48);
        
        // All weights should be equal
        let count = index.component_count();
        let expected_weight = 1.0 / count as f64;
        
        for i in 0..count {
            if let Some(ref comp) = index.components[i] {
                assert!((comp.current_weight - expected_weight).abs() < 0.001);
            }
        }
    }
}
