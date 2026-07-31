//! Linear Algebra Module Root
//! 
//! Provides zero-allocation linear algebra primitives strictly bounded to the 6.5GB RAM limit.

pub mod matrix;
pub mod cholesky;

pub use matrix::{Matrix, Vector, Matrix8x8, Matrix16x16, Matrix32x32, compute_covariance, outer_product};
pub use cholesky::{
    cholesky, solve_cholesky, generate_correlated_samples, 
    log_determinant_cholesky, inverse_cholesky, verify_cholesky,
    make_positive_definite, CholeskyResult,
};

use core::sync::atomic::{AtomicU64, Ordering};

/// Linear algebra operation statistics tracker
#[repr(C, align(64))]
pub struct LinalgStats {
    /// Total matrix multiplications performed
    matmul_count: AtomicU64,
    /// Total Cholesky decompositions performed
    cholesky_count: AtomicU64,
    /// Total solve operations performed
    solve_count: AtomicU64,
    /// Total bytes allocated (should remain near zero for stack-allocated ops)
    bytes_allocated: AtomicU64,
}

impl LinalgStats {
    pub const fn new() -> Self {
        Self {
            matmul_count: AtomicU64::new(0),
            cholesky_count: AtomicU64::new(0),
            solve_count: AtomicU64::new(0),
            bytes_allocated: AtomicU64::new(0),
        }
    }
    
    #[inline]
    pub fn record_matmul(&self) {
        self.matmul_count.fetch_add(1, Ordering::Relaxed);
    }
    
    #[inline]
    pub fn record_cholesky(&self) {
        self.cholesky_count.fetch_add(1, Ordering::Relaxed);
    }
    
    #[inline]
    pub fn record_solve(&self) {
        self.solve_count.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn get_stats(&self) -> LinalgSnapshot {
        LinalgSnapshot {
            matmul_count: self.matmul_count.load(Ordering::Relaxed),
            cholesky_count: self.cholesky_count.load(Ordering::Relaxed),
            solve_count: self.solve_count.load(Ordering::Relaxed),
            bytes_allocated: self.bytes_allocated.load(Ordering::Relaxed),
        }
    }
    
    pub fn reset(&self) {
        self.matmul_count.store(0, Ordering::Relaxed);
        self.cholesky_count.store(0, Ordering::Relaxed);
        self.solve_count.store(0, Ordering::Relaxed);
        self.bytes_allocated.store(0, Ordering::Relaxed);
    }
}

/// Snapshot of linear algebra statistics
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct LinalgSnapshot {
    pub matmul_count: u64,
    pub cholesky_count: u64,
    pub solve_count: u64,
    pub bytes_allocated: u64,
}

impl Default for LinalgStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Global statistics instance (can be accessed from anywhere)
pub static GLOBAL_LINALG_STATS: LinalgStats = LinalgStats::new();

/// Fast dot product for small vectors using unrolled loops
#[inline]
pub fn fast_dot<const N: usize>(a: &[f64; N], b: &[f64; N]) -> f64 {
    let mut sum = 0.0;
    
    // Unroll for common sizes
    if N <= 8 {
        for i in 0..N {
            sum += a[i] * b[i];
        }
    } else {
        // For larger vectors, use SIMD-friendly pattern
        let chunks = N / 4;
        let remainder = N % 4;
        
        for i in 0..chunks {
            let base = i * 4;
            sum += a[base] * b[base];
            sum += a[base + 1] * b[base + 1];
            sum += a[base + 2] * b[base + 2];
            sum += a[base + 3] * b[base + 3];
        }
        
        for i in (chunks * 4)..N {
            sum += a[i] * b[i];
        }
    }
    
    sum
}

/// Fast matrix-vector multiplication
#[inline]
pub fn mat_vec_mul<const N: usize, const M: usize>(
    matrix: &[[f64; M]; N],
    vector: &[f64; M],
) -> [f64; N] {
    let mut result = [0.0; N];
    
    for i in 0..N {
        result[i] = fast_dot(&matrix[i], vector);
    }
    
    result
}

/// Compute portfolio variance given weights and covariance matrix
/// Var(p) = w^T * Cov * w
#[inline]
pub fn portfolio_variance<const N: usize>(
    weights: &[f64; N],
    covariance: &Matrix<N, N>,
) -> f64 {
    let mut variance = 0.0;
    
    for i in 0..N {
        for j in 0..N {
            if let Some(cov_ij) = covariance.get(i, j) {
                variance += weights[i] * cov_ij * weights[j];
            }
        }
    }
    
    variance
}

/// Compute portfolio expected return given weights and asset returns
#[inline]
pub fn portfolio_return<const N: usize>(
    weights: &[f64; N],
    returns: &[f64; N],
) -> f64 {
    fast_dot(weights, returns)
}

/// Sharpe ratio calculator (annualized)
#[inline]
pub fn sharpe_ratio(
    portfolio_return: f64,
    risk_free_rate: f64,
    portfolio_std: f64,
) -> f64 {
    if portfolio_std < 1e-15 {
        return 0.0;
    }
    (portfolio_return - risk_free_rate) / portfolio_std
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fast_dot() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        
        let result = fast_dot(&a, &b);
        assert_eq!(result, 70.0);
    }
    
    #[test]
    fn test_mat_vec_mul() {
        let m = [[1.0, 2.0], [3.0, 4.0]];
        let v = [1.0, 1.0];
        
        let result = mat_vec_mul(&m, &v);
        assert_eq!(result[0], 3.0);
        assert_eq!(result[1], 7.0);
    }
    
    #[test]
    fn test_portfolio_metrics() {
        let weights = [0.5, 0.3, 0.2];
        let returns = [0.10, 0.08, 0.06];
        
        let ret = portfolio_return(&weights, &returns);
        assert!((ret - 0.086).abs() < 1e-10);
        
        // Identity covariance (uncorrelated assets with unit variance)
        let cov = Matrix::<3, 3>::identity();
        let var = portfolio_variance(&weights, &cov);
        
        // Variance should be sum of squared weights
        let expected_var = 0.25 + 0.09 + 0.04;
        assert!((var - expected_var).abs() < 1e-10);
    }
    
    #[test]
    fn test_sharpe_ratio() {
        let sr = sharpe_ratio(0.12, 0.02, 0.15);
        assert!((sr - 0.6666666666666666).abs() < 1e-10);
    }
    
    #[test]
    fn test_stats_tracking() {
        let stats = LinalgStats::new();
        
        stats.record_matmul();
        stats.record_matmul();
        stats.record_cholesky();
        
        let snapshot = stats.get_stats();
        assert_eq!(snapshot.matmul_count, 2);
        assert_eq!(snapshot.cholesky_count, 1);
        assert_eq!(snapshot.solve_count, 0);
        
        stats.reset();
        let snapshot = stats.get_stats();
        assert_eq!(snapshot.matmul_count, 0);
    }
}
