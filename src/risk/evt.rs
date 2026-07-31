//! Generalized Pareto Distribution (GPD) and Peaks-Over-Threshold (POT) modeling
//! for extreme tail risk estimation in high-frequency crypto trading.
//!
//! This module implements numerically stable EVT calculations using Gauss-Legendre
//! quadrature for integration without temporary allocations.

use std::f64::consts::PI;

/// Pre-computed Gauss-Legendre quadrature nodes and weights for n=32
/// These are hardcoded for maximum performance and zero allocation
const GL_NODES: [f64; 32] = [
    -0.9985014260311177, -0.9890877230977301, -0.9709272963697471, -0.9442792282889194,
    -0.9095730194864929, -0.8673582515966902, -0.8183082741945923, -0.7632119349968242,
    -0.7029704793436092, -0.6385831286536479, -0.5711319134638696, -0.5017652133199964,
    -0.4316823951875213, -0.3621165165973998, -0.2942203985991965, -0.2290523470306567,
     0.2290523470306567,  0.2942203985991965,  0.3621165165973998,  0.4316823951875213,
     0.5017652133199964,  0.5711319134638696,  0.6385831286536479,  0.7029704793436092,
     0.7632119349968242,  0.8183082741945923,  0.8673582515966902,  0.9095730194864929,
     0.9442792282889194,  0.9709272963697471,  0.9890877230977301,  0.9985014260311177,
];

const GL_WEIGHTS: [f64; 32] = [
    0.0038711680711789,  0.0089981399886297,  0.0141108195882883,  0.0191860505395392,
    0.0242008585095935,  0.0291325328508407,  0.0339588375603382,  0.0386579947174473,
    0.0432087623871402,  0.0475905220361489,  0.0517833605195293,  0.0557681433351423,
    0.0595265904352347,  0.0630413430046182,  0.0662959307085801,  0.0692748393750779,
    0.0692748393750779,  0.0662959307085801,  0.0630413430046182,  0.0595265904352347,
    0.0557681433351423,  0.0517833605195293,  0.0475905220361489,  0.0432087623871402,
    0.0386579947174473,  0.0339588375603382,  0.0291325328508407,  0.0242008585095935,
    0.0191860505395392,  0.0141108195882883,  0.0089981399886297,  0.0038711680711789,
];

/// Parameters for the Generalized Pareto Distribution
#[derive(Debug, Clone, Copy)]
pub struct GPDParameters {
    /// Shape parameter (xi): determines tail heaviness
    pub xi: f64,
    /// Scale parameter (sigma): must be positive
    pub sigma: f64,
    /// Threshold (u): the level above which we model exceedances
    pub threshold: f64,
}

impl GPDParameters {
    pub fn new(xi: f64, sigma: f64, threshold: f64) -> Option<Self> {
        if sigma <= 0.0 {
            return None;
        }
        Some(GPDParameters { xi, sigma, threshold })
    }
}

/// Extreme Value Theory engine for tail risk modeling
pub struct ExtremeValueTheory {
    params: GPDParameters,
    /// Pre-allocated buffer for numerical integration (zero allocation during runtime)
    integration_buffer: [f64; 32],
}

impl ExtremeValueTheory {
    /// Create a new EVT engine with given GPD parameters
    pub fn new(params: GPDParameters) -> Self {
        ExtremeValueTheory {
            params,
            integration_buffer: [0.0; 32],
        }
    }

    /// Update GPD parameters (called when POT re-estimation occurs)
    #[inline]
    pub fn update_parameters(&mut self, params: GPDParameters) {
        self.params = params;
    }

    /// Get current GPD parameters
    #[inline]
    pub fn parameters(&self) -> &GPDParameters {
        &self.params
    }

    /// Compute the GPD survival function P(X > x) for x > threshold
    /// Uses numerically stable computation for small xi values
    #[inline]
    pub fn survival_function(&self, x: f64) -> f64 {
        if x <= self.params.threshold {
            return 1.0;
        }
        
        let z = (x - self.params.threshold) / self.params.sigma;
        let xi = self.params.xi;
        
        // Handle xi close to zero (exponential limit case)
        if xi.abs() < 1e-10 {
            return (-z).exp();
        }
        
        let base = 1.0 + xi * z;
        if base <= 0.0 {
            return 0.0; // Beyond support
        }
        
        base.powf(-1.0 / xi)
    }

    /// Compute the GPD cumulative distribution function
    #[inline]
    pub fn cdf(&self, x: f64) -> f64 {
        1.0 - self.survival_function(x)
    }

    /// Compute the GPD probability density function
    #[inline]
    pub fn pdf(&self, x: f64) -> f64 {
        if x <= self.params.threshold {
            return 0.0;
        }
        
        let z = (x - self.params.threshold) / self.params.sigma;
        let xi = self.params.xi;
        let sigma = self.params.sigma;
        
        if xi.abs() < 1e-10 {
            return (-z).exp() / sigma;
        }
        
        let base = 1.0 + xi * z;
        if base <= 0.0 {
            return 0.0;
        }
        
        base.powf(-1.0 / xi - 1.0) / sigma
    }

    /// Compute Expected Shortfall (ES) at confidence level alpha using Gauss-Legendre quadrature
    /// ES_alpha = E[X | X > VaR_alpha] for tail beyond threshold
    /// 
    /// This uses pre-allocated buffers and hardcoded quadrature nodes for zero allocation
    pub fn expected_shortfall(&self, alpha: f64) -> f64 {
        // VaR at level alpha (quantile of the tail distribution)
        let var = self.var(alpha);
        
        if var <= self.params.threshold {
            // If VaR is below threshold, use analytical formula
            return self.analytical_es(alpha);
        }
        
        // ES = (1 / (1-alpha)) * integral from var to infinity of x * f(x) dx
        // Transform to finite interval using t = 1/(1+x) substitution
        // Then apply Gauss-Legendre quadrature
        
        let xi = self.params.xi;
        let sigma = self.params.sigma;
        let u = self.params.threshold;
        
        // For xi < 1, ES has closed form; otherwise use numerical integration
        if xi < 1.0 && xi > -1.0 {
            // Analytical formula for valid range
            let z_var = (var - u) / sigma;
            let base = 1.0 + xi * z_var;
            
            if base <= 0.0 {
                return var;
            }
            
            // ES = var + sigma * (1 - xi).powf(-1) * base
            // Simplified: ES = (var + sigma / (1 - xi) * (1 + xi * z_var)) / (1 - alpha factor)
            return var + (sigma + xi * (var - u)) / (1.0 - xi);
        }
        
        // Numerical integration for heavy tails (xi >= 1)
        // Transform: integrate from var to M (large finite value)
        let m = u + 100.0 * sigma; // Truncate at reasonable upper bound
        
        // Map [-1, 1] to [var, m]
        let half_range = (m - var) / 2.0;
        let mid = (m + var) / 2.0;
        
        let mut integral = 0.0;
        for i in 0..32 {
            let x = mid + half_range * GL_NODES[i];
            let fx = x * self.pdf(x);
            integral += GL_WEIGHTS[i] * fx;
        }
        
        integral *= half_range;
        integral / (1.0 - alpha)
    }

    /// Analytical Expected Shortfall for xi < 1
    fn analytical_es(&self, alpha: f64) -> f64 {
        let xi = self.params.xi;
        let sigma = self.params.sigma;
        let u = self.params.threshold;
        
        if xi >= 1.0 || xi <= -1.0 {
            return f64::INFINITY;
        }
        
        let var = self.var(alpha);
        let survival = 1.0 - alpha;
        
        // ES = (u - sigma/xi) + (sigma/xi) * (1/(1-xi)) * survival^(-xi)
        // Simplified for GPD
        if xi.abs() < 1e-10 {
            // Exponential case
            return var + sigma;
        }
        
        let term = survival.powf(-xi);
        u + (sigma / xi) * (term / (1.0 - xi) - 1.0)
    }

    /// Compute Value at Risk (VaR) at confidence level alpha
    /// Returns the quantile such that P(X > VaR) = 1 - alpha
    #[inline]
    pub fn var(&self, alpha: f64) -> f64 {
        let xi = self.params.xi;
        let sigma = self.params.sigma;
        let u = self.params.threshold;
        let survival = 1.0 - alpha;
        
        if xi.abs() < 1e-10 {
            // Exponential limit
            return u + sigma * (-survival.ln());
        }
        
        u + (sigma / xi) * (survival.powf(-xi) - 1.0)
    }

    /// Peaks-Over-Threshold: Estimate tail probability for extreme events
    /// P(X > x) for x significantly above threshold
    #[inline]
    pub fn pot_tail_probability(&self, x: f64) -> f64 {
        self.survival_function(x)
    }

    /// Compute conditional excess E[X - u | X > u] (mean excess function)
    #[inline]
    pub fn mean_excess(&self) -> f64 {
        let xi = self.params.xi;
        let sigma = self.params.sigma;
        
        if xi >= 1.0 {
            return f64::INFINITY; // Infinite mean excess
        }
        
        sigma / (1.0 - xi)
    }

    /// Calculate tail index for extreme event classification
    /// Higher values indicate heavier tails (more extreme events)
    #[inline]
    pub fn tail_index(&self) -> f64 {
        1.0 / self.params.xi.max(1e-10)
    }

    /// Stress test: compute ES under parameter perturbation
    pub fn stress_test(&self, xi_shock: f64, sigma_shock: f64, alpha: f64) -> f64 {
        let stressed_xi = self.params.xi * (1.0 + xi_shock);
        let stressed_sigma = self.params.sigma * (1.0 + sigma_shock);
        
        if let Some(stressed_params) = GPDParameters::new(
            stressed_xi,
            stressed_sigma,
            self.params.threshold,
        ) {
            let stressed_evt = ExtremeValueTheory::new(stressed_params);
            stressed_evt.expected_shortfall(alpha)
        } else {
            f64::INFINITY
        }
    }

    /// Compute the Hill estimator for tail index from sorted exceedances
    /// This is a static method for initial parameter estimation
    pub fn hill_estimator(exceedances: &[f64], k: usize) -> f64 {
        if k == 0 || exceedances.len() < k {
            return f64::NAN;
        }
        
        let log_sum: f64 = exceedances[..k]
            .iter()
            .map(|&x| x.ln())
            .sum();
        
        log_sum / k as f64
    }

    /// Maximum Likelihood Estimation score (negative log-likelihood)
    /// Used for parameter optimization (caller provides optimizer)
    pub fn neg_log_likelihood(&self, data: &[f64]) -> f64 {
        let mut nll = 0.0;
        let xi = self.params.xi;
        let sigma = self.params.sigma;
        let u = self.params.threshold;
        
        for &x in data {
            if x <= u {
                continue;
            }
            
            let z = (x - u) / sigma;
            let base = 1.0 + xi * z;
            
            if base <= 0.0 {
                return f64::INFINITY; // Invalid parameters for this data
            }
            
            let log_pdf = -sigma.ln() - (1.0 / xi + 1.0) * base.ln();
            nll -= log_pdf;
        }
        
        nll
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpd_survival() {
        let params = GPDParameters::new(0.3, 1.0, 0.0).unwrap();
        let evt = ExtremeValueTheory::new(params);
        
        // Survival should decrease as x increases
        assert!(evt.survival_function(1.0) > evt.survival_function(2.0));
        assert!(evt.survival_function(2.0) > evt.survival_function(5.0));
    }

    #[test]
    fn test_expected_shortfall_monotonicity() {
        let params = GPDParameters::new(0.2, 1.0, 0.0).unwrap();
        let evt = ExtremeValueTheory::new(params);
        
        // ES should increase as alpha increases (higher confidence = more extreme)
        let es_95 = evt.expected_shortfall(0.95);
        let es_99 = evt.expected_shortfall(0.99);
        let es_99_9 = evt.expected_shortfall(0.999);
        
        assert!(es_95 < es_99);
        assert!(es_99 < es_99_9);
    }

    #[test]
    fn test_hill_estimator() {
        // Generate some fake exceedances (sorted descending)
        let exceedances: Vec<f64> = (1..=100).map(|i| 1.0 / i as f64).collect();
        let hill = ExtremeValueTheory::hill_estimator(&exceedances, 20);
        assert!(hill.is_finite());
    }
}
