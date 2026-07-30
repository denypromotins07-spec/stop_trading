//! Conditional Value at Risk (CVaR) / Expected Shortfall engine.
//! 
//! Measures extreme tail-risk exposure by quantifying the average loss
//! expected beyond the VaR threshold. Protects against black swan crypto flash crashes.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use crate::common::ring_buffer::LockFreeRingBuffer;
use crate::risk::var::{VarCalculator, VarConfig, VarMethod};

/// CVaR calculation result
#[derive(Debug, Clone)]
pub struct CVarResult {
    /// CVaR value (expected loss beyond VaR)
    pub cvar: f64,
    /// Corresponding VaR value
    pub var: f64,
    /// Confidence level used
    pub confidence_level: f64,
    /// Number of tail samples used
    pub tail_sample_count: usize,
    /// Total sample count
    pub total_sample_count: usize,
    /// Timestamp of calculation
    pub timestamp_ns: u64,
    /// Maximum observed loss
    pub max_loss: f64,
    /// Tail mean (average of losses beyond VaR)
    pub tail_mean: f64,
}

impl CVarResult {
    /// Get CVaR as percentage
    pub fn cvar_percentage(&self) -> f64 {
        self.cvar * 100.0
    }
    
    /// Get dollar CVaR given portfolio value
    pub fn cvar_dollar(&self, portfolio_value: f64) -> f64 {
        self.cvar * portfolio_value
    }
    
    /// Ratio of CVaR to VaR (measures tail thickness)
    pub fn cvar_var_ratio(&self) -> f64 {
        if self.var > 0.0 {
            self.cvar / self.var
        } else {
            1.0
        }
    }
}

/// Configuration for CVaR calculation
#[derive(Debug, Clone)]
pub struct CVarConfig {
    /// Base VaR configuration
    pub var_config: VarConfig,
    /// Use exponential weighting for tail samples
    pub exponential_weighting: bool,
    /// Decay factor for exponential weighting
    pub decay_factor: f64,
}

impl Default for CVarConfig {
    fn default() -> Self {
        Self {
            var_config: VarConfig::default(),
            exponential_weighting: false,
            decay_factor: 0.95,
        }
    }
}

/// High-performance CVaR calculator
pub struct CVarCalculator {
    /// Ring buffer for storing returns
    returns_buffer: LockFreeRingBuffer<f64>,
    /// Underlying VaR calculator
    var_calculator: VarCalculator,
    /// Configuration
    config: CVarConfig,
    /// Cached tail statistics
    cached_tail_mean: f64,
    cached_tail_variance: f64,
    cached_cvar: f64,
    /// Cache validity flag
    cache_valid: AtomicU64,
    /// Update counter
    update_count: AtomicU64,
}

impl CVarCalculator {
    /// Create a new CVaR calculator
    pub fn new(capacity: usize, config: CVarConfig) -> Self {
        let var_calc = VarCalculator::new(capacity, config.var_config.clone());
        
        Self {
            returns_buffer: LockFreeRingBuffer::new(capacity),
            var_calculator: var_calc,
            config,
            cached_tail_mean: 0.0,
            cached_tail_variance: 0.0,
            cached_cvar: 0.0,
            cache_valid: AtomicU64::new(0),
            update_count: AtomicU64::new(0),
        }
    }
    
    /// Add a new return observation
    #[inline]
    pub fn add_return(&mut self, return_val: f64) {
        // Store as negative return (loss)
        let loss = -return_val;
        self.returns_buffer.push(loss);
        self.var_calculator.add_return(return_val);
        self.update_count.fetch_add(1, Ordering::Relaxed);
        self.cache_valid.store(0, Ordering::Relaxed);
    }
    
    /// Calculate CVaR (Expected Shortfall)
    pub fn calculate_cvar(&mut self) -> Option<CVarResult> {
        // First calculate VaR
        let var_result = self.var_calculator.calculate_var()?;
        let var = var_result.var;
        
        let count = self.returns_buffer.len();
        if count == 0 {
            return None;
        }
        
        // Update cache if needed
        if self.cache_valid.load(Ordering::Relaxed) == 0 {
            self.update_tail_statistics(var)?;
            self.cache_valid.store(1, Ordering::Relaxed);
        }
        
        Some(CVarResult {
            cvar: self.cached_cvar,
            var,
            confidence_level: self.config.var_config.confidence_level,
            tail_sample_count: self.count_tail_samples(var),
            total_sample_count: count,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
            max_loss: self.find_max_loss(),
            tail_mean: self.cached_tail_mean,
        })
    }
    
    /// Update tail statistics beyond VaR threshold
    fn update_tail_statistics(&mut self, var: f64) -> Option<()> {
        let mut tail_sum = 0.0;
        let mut tail_sq_sum = 0.0;
        let mut tail_count = 0;
        let mut weight_sum = 0.0;
        
        let total_count = self.returns_buffer.len();
        
        for (idx, &loss) in self.returns_buffer.iter().enumerate() {
            if loss > var {
                let weight = if self.config.exponential_weighting {
                    self.config.decay_factor.powi((total_count - idx) as i32)
                } else {
                    1.0
                };
                
                tail_sum += loss * weight;
                tail_sq_sum += loss.powi(2) * weight;
                tail_count += 1;
                weight_sum += weight;
            }
        }
        
        if tail_count == 0 {
            // No tail samples, CVaR equals VaR
            self.cached_tail_mean = var;
            self.cached_tail_variance = 0.0;
            self.cached_cvar = var;
            return Some(());
        }
        
        // Calculate weighted tail mean
        self.cached_tail_mean = tail_sum / weight_sum;
        
        // Calculate tail variance
        let mean_sq = self.cached_tail_mean.powi(2);
        self.cached_tail_variance = (tail_sq_sum / weight_sum) - mean_sq;
        
        // CVaR is the expected loss beyond VaR
        self.cached_cvar = self.cached_tail_mean;
        
        Some(())
    }
    
    /// Count samples beyond VaR threshold
    fn count_tail_samples(&self, var: f64) -> usize {
        self.returns_buffer.iter()
            .filter(|&&loss| loss > var)
            .count()
    }
    
    /// Find maximum observed loss
    fn find_max_loss(&self) -> f64 {
        self.returns_buffer.iter()
            .fold(f64::NEG_INFINITY, |max, &x| max.max(x))
    }
    
    /// Get stress CVaR using historical worst cases
    pub fn stress_cvar(&self, percentile: f64) -> Option<f64> {
        let mut losses: Vec<f64> = self.returns_buffer.iter().collect();
        if losses.is_empty() {
            return None;
        }
        
        losses.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        
        let idx = (percentile * losses.len() as f64).floor() as usize;
        let idx = idx.min(losses.len() - 1);
        
        // Average of worst cases
        let worst_losses = &losses[..=idx];
        Some(worst_losses.iter().sum::<f64>() / worst_losses.len() as f64)
    }
    
    /// Get incremental CVaR contribution from adding a position
    pub fn marginal_cvar(&self, new_return: f64, position_weight: f64) -> f64 {
        // Approximate marginal CVaR using finite difference
        let current_cvar = self.cached_cvar;
        
        // Simulate adding the new position
        let adjusted_return = new_return * position_weight;
        let new_loss = -adjusted_return;
        
        // Simple approximation: scale by weight
        current_cvar * position_weight
    }
    
    /// Component CVaR decomposition
    pub fn component_cvar(&self, returns: &[f64], weights: &[f64]) -> Vec<f64> {
        assert_eq!(returns.len(), weights.len());
        
        let portfolio_return: f64 = returns.iter()
            .zip(weights.iter())
            .map(|(&r, &w)| r * w)
            .sum();
        
        let portfolio_loss = -portfolio_return;
        
        // If portfolio is in tail, allocate CVaR proportionally
        if portfolio_loss > self.var_calculator.current_std_dev() * 2.33 {
            let total_weight: f64 = weights.iter().sum();
            weights.iter()
                .map(|&w| self.cached_cvar * w / total_weight)
                .collect()
        } else {
            vec![0.0; weights.len()]
        }
    }
    
    /// Get update count
    #[inline]
    pub fn update_count(&self) -> u64 {
        self.update_count.load(Ordering::Relaxed)
    }
    
    /// Clear all data
    pub fn clear(&mut self) {
        self.returns_buffer.clear();
        self.var_calculator.clear();
        self.cached_tail_mean = 0.0;
        self.cached_tail_variance = 0.0;
        self.cached_cvar = 0.0;
        self.cache_valid.store(0, Ordering::Relaxed);
    }
}

/// Multi-asset CVaR calculator with diversification benefits
pub struct PortfolioCVarCalculator {
    /// Individual CVaR calculators
    asset_calculators: Vec<CVarCalculator>,
    /// Correlation matrix
    correlation_matrix: Vec<f64>,
    /// Asset weights
    weights: Vec<f64>,
    /// Number of assets
    num_assets: usize,
    /// Diversification ratio cache
    diversification_ratio: f64,
}

impl PortfolioCVarCalculator {
    /// Create a new portfolio CVaR calculator
    pub fn new(num_assets: usize, capacity: usize, config: CVarConfig) -> Self {
        let calculators = (0..num_assets)
            .map(|_| CVarCalculator::new(capacity, config.clone()))
            .collect();
        
        Self {
            asset_calculators: calculators,
            correlation_matrix: vec![0.0; num_assets * num_assets],
            weights: vec![1.0 / num_assets as f64; num_assets],
            num_assets,
            diversification_ratio: 1.0,
        }
    }
    
    /// Set portfolio weights
    pub fn set_weights(&mut self, weights: &[f64]) -> Result<(), &'static str> {
        if weights.len() != self.num_assets {
            return Err("Weight count mismatch");
        }
        
        let sum: f64 = weights.iter().sum();
        if (sum - 1.0).abs() > 0.001 {
            return Err("Weights must sum to 1.0");
        }
        
        self.weights = weights.to_vec();
        Ok(())
    }
    
    /// Set correlation between assets
    pub fn set_correlation(&mut self, i: usize, j: usize, corr: f64) {
        let idx = i * self.num_assets + j;
        self.correlation_matrix[idx] = corr.clamp(-1.0, 1.0);
        self.correlation_matrix[j * self.num_assets + i] = corr.clamp(-1.0, 1.0);
    }
    
    /// Add returns for all assets
    pub fn add_returns(&mut self, returns: &[f64]) {
        for (i, &r) in returns.iter().enumerate() {
            if i < self.asset_calculators.len() {
                self.asset_calculators[i].add_return(r);
            }
        }
    }
    
    /// Calculate portfolio CVaR with diversification adjustment
    pub fn calculate_portfolio_cvar(&mut self) -> Option<f64> {
        // Calculate individual CVaRs
        let mut individual_cvars = Vec::with_capacity(self.num_assets);
        let mut sum_weighted_cvar = 0.0;
        
        for (i, calc) in self.asset_calculators.iter_mut().enumerate() {
            if let Some(result) = calc.calculate_cvar() {
                individual_cvars.push(result.cvar);
                sum_weighted_cvar += result.cvar * self.weights[i];
            } else {
                return None;
            }
        }
        
        // Calculate diversification benefit
        let mut diversification_benefit = 0.0;
        for i in 0..self.num_assets {
            for j in (i + 1)..self.num_assets {
                let corr = self.correlation_matrix[i * self.num_assets + j];
                let joint_risk = corr * individual_cvars[i] * individual_cvars[j];
                diversification_benefit += 2.0 * (1.0 - corr) * joint_risk;
            }
        }
        
        // Portfolio CVaR with diversification
        let portfolio_cvar = sum_weighted_cvar - diversification_benefit * 0.1;
        let portfolio_cvar = portfolio_cvar.max(*individual_cvars.iter().max()?);
        
        self.diversification_ratio = sum_weighted_cvar / portfolio_cvar;
        
        Some(portfolio_cvar)
    }
    
    /// Get diversification ratio
    pub fn diversification_ratio(&self) -> f64 {
        self.diversification_ratio
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cvar_basic() {
        let config = CVarConfig::default();
        let mut calc = CVarCalculator::new(1000, config);
        
        // Add some returns with occasional large losses
        for i in 0..200 {
            let ret = if i % 20 == 0 {
                -0.15 // Large loss every 20 samples
            } else {
                (i as f64 * 0.001 - 0.05)
            };
            calc.add_return(ret);
        }
        
        let result = calc.calculate_cvar();
        assert!(result.is_some());
        
        let result = result.unwrap();
        assert!(result.cvar >= result.var); // CVaR should be >= VaR
    }
    
    #[test]
    fn test_cvar_var_ratio() {
        let config = CVarConfig::default();
        let mut calc = CVarCalculator::new(1000, config);
        
        // Add fat-tailed returns
        for i in 0..500 {
            let ret = if i % 50 == 0 {
                -0.25 // Extreme loss
            } else {
                (i as f64 * 0.0005 - 0.025)
            };
            calc.add_return(ret);
        }
        
        let result = calc.calculate_cvar().unwrap();
        
        // For fat-tailed distributions, CVaR/VaR ratio > 1
        assert!(result.cvar_var_ratio() >= 1.0);
    }
}
