//! Copula Engine for modeling non-linear tail dependence between multi-asset portfolios.
//!
//! Implements Gaussian and Student-t Copulas with Gauss-Legendre quadrature for
//! computing joint tail probabilities without temporary allocations.

use std::f64::consts::{PI, SQRT_2};

/// Pre-computed Gauss-Legendre quadrature nodes and weights for n=16 (bivariate integration)
const GL_NODES_16: [f64; 16] = [
    -0.9894009349916499, -0.9445750230732325, -0.8656312023878317, -0.7554044083550030,
    -0.6178762444026437, -0.4580167776572273, -0.2816035507792589, -0.0950125098376374,
     0.0950125098376374,  0.2816035507792589,  0.4580167776572273,  0.6178762444026437,
     0.7554044083550030,  0.8656312023878317,  0.9445750230732325,  0.9894009349916499,
];

const GL_WEIGHTS_16: [f64; 16] = [
    0.0271524594117541,  0.0622535239386479,  0.0951585116824928,  0.1246289712555339,
    0.1495959888165767,  0.1691565193950025,  0.1826080950586646,  0.1894506104550685,
    0.1894506104550685,  0.1826080950586646,  0.1691565193950025,  0.1495959888165767,
    0.1246289712555339,  0.0951585116824928,  0.0622535239386479,  0.0271524594117541,
];

/// Standard normal CDF approximation (Abramowitz & Stegun)
#[inline]
fn phi(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d = 0.3989422804014327 * (-x * x / 2.0).exp();
    let p = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    if x > 0.0 {
        1.0 - d * p
    } else {
        d * p
    }
}

/// Inverse standard normal CDF (Rational approximation)
#[inline]
fn phi_inv(p: f64) -> f64 {
    if p <= 0.0 || p >= 1.0 {
        return f64::NAN;
    }
    
    // Rational approximation for central region
    if p > 0.0 && p < 1.0 {
        let a = [
            -3.969683028665376e+01,
             2.209460984245205e+02,
            -2.759285104469687e+02,
             1.383577518672690e+02,
            -3.066479806614716e+01,
             2.506628277459239e+00,
        ];
        let b = [
            -5.447609879822406e+01,
             1.615858368580409e+02,
            -1.556989798598866e+02,
             6.680131188771972e+01,
            -1.328068155288572e+01,
        ];
        let c = [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e+00,
            -2.549732539343734e+00,
             4.374664141464968e+00,
             2.938163982698783e+00,
        ];
        let d = [
             7.784695709041462e-03,
             3.224671290700398e-01,
             2.445134137142996e+00,
             3.754408661907416e+00,
        ];
        
        let p_low = 0.02425;
        let p_high = 1.0 - p_low;
        
        if p < p_low {
            let q = (-2.0 * p.ln()).sqrt();
            return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                   ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
        } else if p <= p_high {
            let q = p - 0.5;
            let r = q * q;
            return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
                   (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
        } else {
            let q = (-2.0 * (1.0 - p).ln()).sqrt();
            return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                    ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
        }
    }
    
    f64::NAN
}

/// Bivariate normal CDF using Gauss-Legendre quadrature
/// Computes P(Z1 <= a, Z2 <= b) with correlation rho
#[inline]
fn bivariate_normal_cdf(a: f64, b: f64, rho: f64) -> f64 {
    if rho.abs() >= 1.0 {
        // Perfect correlation case
        if rho > 0.0 {
            return phi(a.min(b));
        } else {
            return (phi(a) + phi(b) - 1.0).max(0.0);
        }
    }
    
    let sqrt_one_rho2 = (1.0 - rho * rho).sqrt();
    let mut result = 0.0;
    
    // Single integral transformation for bivariate normal
    for i in 0..16 {
        let z = GL_NODES_16[i];
        let az = a - rho * z;
        let bz = b - rho * z;
        
        let integrand = phi(az / sqrt_one_rho2) * phi(bz / sqrt_one_rho2);
        result += GL_WEIGHTS_16[i] * integrand;
    }
    
    // Scale by the transformation factor
    result * sqrt_one_rho2 / SQRT_2
}

/// Trait defining copula operations
pub trait Copula {
    /// Compute the copula C(u1, u2, ..., un) value
    fn evaluate(&self, u: &[f64]) -> f64;
    
    /// Compute lower tail dependence coefficient lambda_L
    fn lower_tail_dependence(&self) -> f64;
    
    /// Compute upper tail dependence coefficient lambda_U
    fn upper_tail_dependence(&self) -> f64;
    
    /// Simulate joint default probability (all assets below threshold)
    fn joint_tail_probability(&self, thresholds: &[f64]) -> f64;
}

/// Gaussian Copula for modeling symmetric dependence
pub struct GaussianCopula {
    /// Correlation matrix (stored as flat array for cache efficiency)
    /// For bivariate: just the correlation coefficient rho
    pub rho: f64,
    /// Pre-computed sqrt(1 - rho^2)
    sqrt_one_rho2: f64,
}

impl GaussianCopula {
    pub fn new(rho: f64) -> Option<Self> {
        if rho <= -1.0 || rho >= 1.0 {
            return None;
        }
        Some(GaussianCopula {
            rho,
            sqrt_one_rho2: (1.0 - rho * rho).sqrt(),
        })
    }
    
    /// Create from empirical correlation using Kendall's tau
    pub fn from_kendall_tau(tau: f64) -> Option<Self> {
        // rho = sin(tau * PI / 2)
        let rho = (tau * PI / 2.0).sin();
        Self::new(rho)
    }
}

impl Copula for GaussianCopula {
    fn evaluate(&self, u: &[f64]) -> f64 {
        if u.len() != 2 {
            // For now, only bivariate is fully implemented
            // Extend to multivariate with Cholesky decomposition
            return f64::NAN;
        }
        
        let u1 = u[0];
        let u2 = u[1];
        
        if u1 <= 0.0 || u1 >= 1.0 || u2 <= 0.0 || u2 >= 1.0 {
            return 0.0;
        }
        
        // Transform to normal space
        let x1 = phi_inv(u1);
        let x2 = phi_inv(u2);
        
        // Bivariate normal CDF
        bivariate_normal_cdf(x1, x2, self.rho)
    }
    
    /// Gaussian copula has zero tail dependence
    fn lower_tail_dependence(&self) -> f64 {
        0.0
    }
    
    fn upper_tail_dependence(&self) -> f64 {
        0.0
    }
    
    fn joint_tail_probability(&self, thresholds: &[f64]) -> f64 {
        if thresholds.len() != 2 {
            return f64::NAN;
        }
        
        let u1 = thresholds[0];
        let u2 = thresholds[1];
        
        self.evaluate(&[u1, u2])
    }
}

/// Student-t Copula for modeling tail dependence (symmetric)
pub struct StudentTCopula {
    /// Correlation coefficient
    pub rho: f64,
    /// Degrees of freedom (nu): lower values = heavier tails
    pub nu: f64,
    /// Pre-computed constants
    sqrt_one_rho2: f64,
}

impl StudentTCopula {
    pub fn new(rho: f64, nu: f64) -> Option<Self> {
        if rho <= -1.0 || rho >= 1.0 || nu <= 0.0 {
            return None;
        }
        Some(StudentTCopula {
            rho,
            nu,
            sqrt_one_rho2: (1.0 - rho * rho).sqrt(),
        })
    }
    
    /// Create from empirical data using method of moments
    pub fn from_tail_dependence(lambda: f64, nu: f64) -> Option<Self> {
        // For t-copula: lambda_L = lambda_U = 2 * t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho)))
        // Inverse relationship to find rho from lambda
        // Simplified: use approximation
        if lambda < 0.0 || lambda > 1.0 {
            return None;
        }
        
        // Approximate inversion
        let rho = 1.0 - (lambda.powf(2.0 / nu) * nu) / (nu + 1.0);
        let rho = rho.max(-0.99).min(0.99);
        
        Self::new(rho, nu)
    }
}

/// Student-t CDF approximation
#[inline]
fn student_t_cdf(x: f64, nu: f64) -> f64 {
    if nu <= 0.0 {
        return f64::NAN;
    }
    
    // Use regularized incomplete beta function approximation
    // For large nu, approaches normal
    if nu > 100.0 {
        return phi(x);
    }
    
    // Simple approximation using series expansion
    let t2 = x * x;
    let y = t2 / (nu + t2);
    
    // Incomplete beta approximation
    let mut result = 0.5;
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    
    // Series expansion for incomplete beta
    let mut term = 1.0;
    let mut sum = 1.0;
    for k in 1..=50 {
        term *= y * (k as f64 - 0.5) / (k as f64 + nu / 2.0 - 1.0);
        sum += term;
        if term.abs() < 1e-15 {
            break;
        }
    }
    
    let beta_coef = (nu * PI).sqrt() / (nu * beta_func(nu / 2.0, 0.5));
    result += sign * 0.5 * beta_coef * x * sum / (1.0 + t2 / nu).powf((nu + 1.0) / 2.0);
    
    result.max(0.0).min(1.0)
}

/// Beta function approximation
#[inline]
fn beta_func(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Log-gamma function approximation (Lanczos)
#[inline]
fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }
    
    let g = 7.0;
    let c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    
    if x < 0.5 {
        return (PI / (x * (PI * x).sin())).ln() - ln_gamma(1.0 - x);
    }
    
    let x = x - 1.0;
    let mut y = c[0];
    for i in 1..c.len() {
        y += c[i] / (x + i as f64);
    }
    
    let t = x + g + 0.5;
    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + y.ln()
}

/// Inverse Student-t CDF
#[inline]
fn student_t_inv(p: f64, nu: f64) -> f64 {
    if p <= 0.0 || p >= 1.0 || nu <= 0.0 {
        return f64::NAN;
    }
    
    // Use Newton-Raphson iteration starting from normal approximation
    let mut x = phi_inv(p);
    
    for _ in 0..10 {
        let fx = student_t_cdf(x, nu) - p;
        if fx.abs() < 1e-12 {
            break;
        }
        
        // Derivative of t-CDF is t-PDF
        let pdf = (1.0 + x * x / nu).powf(-(nu + 1.0) / 2.0) 
                  / (nu.sqrt() * beta_func(nu / 2.0, 0.5));
        
        x -= fx / pdf;
    }
    
    x
}

impl Copula for StudentTCopula {
    fn evaluate(&self, u: &[f64]) -> f64 {
        if u.len() != 2 {
            return f64::NAN;
        }
        
        let u1 = u[0].max(1e-15).min(1.0 - 1e-15);
        let u2 = u[1].max(1e-15).min(1.0 - 1e-15);
        
        // Transform to t-space
        let t1 = student_t_inv(u1, self.nu);
        let t2 = student_t_inv(u2, self.nu);
        
        // Bivariate t-CDF approximation using numerical integration
        // Using single integral representation
        let nu = self.nu;
        let rho = self.rho;
        let sqrt_one_rho2 = self.sqrt_one_rho2;
        
        // Integral over mixing variable for t-distribution
        let mut result = 0.0;
        
        // Gauss-Legendre quadrature over transformed domain
        for i in 0..16 {
            let w = GL_NODES_16[i];
            // Transform from [-1,1] to [0, inf) for chi-squared mixing
            let s = ((w + 1.0) / 2.0).max(1e-10);
            let scale = (nu * s).sqrt();
            
            let cond1 = phi((t1 * scale.sqrt()) / sqrt_one_rho2);
            let cond2 = phi((t2 * scale.sqrt()) / sqrt_one_rho2);
            
            result += GL_WEIGHTS_16[i] * cond1 * cond2;
        }
        
        result.max(0.0).min(1.0)
    }
    
    /// Student-t copula has symmetric tail dependence
    fn lower_tail_dependence(&self) -> f64 {
        let nu = self.nu;
        let rho = self.rho;
        
        if rho >= 1.0 {
            return 1.0;
        }
        
        // lambda_L = 2 * t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho)))
        let arg = -((nu + 1.0) * (1.0 - rho) / (1.0 + rho)).sqrt();
        2.0 * student_t_cdf(arg, nu + 1.0)
    }
    
    fn upper_tail_dependence(&self) -> f64 {
        // Symmetric for t-copula
        self.lower_tail_dependence()
    }
    
    fn joint_tail_probability(&self, thresholds: &[f64]) -> f64 {
        if thresholds.len() != 2 {
            return f64::NAN;
        }
        
        self.evaluate(thresholds)
    }
}

/// Multi-asset portfolio tail risk calculator
pub struct PortfolioTailRisk {
    /// Vector of pairwise copulas for N assets
    copulas: Vec<Box<dyn Copula + Send + Sync>>,
    /// Asset indices for each copula
    pairs: Vec<(usize, usize)>,
}

impl PortfolioTailRisk {
    pub fn new() -> Self {
        PortfolioTailRisk {
            copulas: Vec::new(),
            pairs: Vec::new(),
        }
    }
    
    /// Add a pairwise copula relationship
    pub fn add_copula(&mut self, i: usize, j: usize, copula: Box<dyn Copula + Send + Sync>) {
        self.copulas.push(copula);
        self.pairs.push((i, j));
    }
    
    /// Calculate probability of simultaneous flash crash across all monitored pairs
    /// thresholds[i] = P(asset_i < crash_level_i)
    pub fn simultaneous_crash_probability(&self, thresholds: &[f64]) -> f64 {
        if thresholds.is_empty() {
            return f64::NAN;
        }
        
        // For simplicity, use Fréchet-Hoeffding bounds approximation
        // More accurate: use vine copulas or full multivariate integration
        
        let mut min_prob = f64::INFINITY;
        let mut max_prob = 0.0;
        
        for (copula, &(i, j)) in self.copulas.iter().zip(self.pairs.iter()) {
            if i >= thresholds.len() || j >= thresholds.len() {
                continue;
            }
            
            let pair_prob = copula.joint_tail_probability(&[thresholds[i], thresholds[j]]);
            min_prob = min_prob.min(pair_prob);
            max_prob = max_prob.max(pair_prob);
        }
        
        // Conservative estimate: use maximum pairwise probability
        // as lower bound on systemic risk
        max_prob
    }
    
    /// Calculate diversification failure index
    /// Higher values indicate more correlated tail risk (less diversification benefit)
    pub fn diversification_failure_index(&self, stress_level: f64) -> f64 {
        let thresholds = vec![stress_level; 10]; // Assume 10 assets
        
        let independent_prob = stress_level.powi(2); // If independent
        let actual_prob = self.simultaneous_crash_probability(&thresholds);
        
        if independent_prob < 1e-15 {
            return 0.0;
        }
        
        (actual_prob / independent_prob).min(1e6) // Cap for numerical stability
    }
}

impl Default for PortfolioTailRisk {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_copula_creation() {
        let copula = GaussianCopula::new(0.5).unwrap();
        assert_eq!(copula.rho, 0.5);
        assert!(copula.lower_tail_dependence().abs() < 1e-10);
    }

    #[test]
    fn test_student_t_tail_dependence() {
        let copula = StudentTCopula::new(0.5, 4.0).unwrap();
        let lambda_l = copula.lower_tail_dependence();
        let lambda_u = copula.upper_tail_dependence();
        
        // t-copula should have positive tail dependence
        assert!(lambda_l > 0.0);
        assert!(lambda_u > 0.0);
        // Symmetric
        assert!((lambda_l - lambda_u).abs() < 1e-10);
    }

    #[test]
    fn test_portfolio_risk() {
        let mut portfolio = PortfolioTailRisk::new();
        
        // Add BTC-ETH correlation
        if let Some(copula) = StudentTCopula::new(0.7, 5.0) {
            portfolio.add_copula(0, 1, Box::new(copula));
        }
        
        // Add ETH-SOL correlation
        if let Some(copula) = StudentTCopula::new(0.6, 4.0) {
            portfolio.add_copula(1, 2, Box::new(copula));
        }
        
        // Stress scenario: 1% crash probability per asset
        let thresholds = vec![0.01, 0.01, 0.01];
        let joint_prob = portfolio.simultaneous_crash_probability(&thresholds);
        
        assert!(joint_prob.is_finite());
        assert!(joint_prob > 0.0);
    }
}
