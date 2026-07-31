//! Cholesky Decomposition
//! 
//! Highly optimized Cholesky decomposition for generating correlated random variables.
//! Accelerates Monte Carlo simulations and Black-Litterman models without heavy external libraries.
//! Uses in-place decomposition for cache efficiency.

use crate::linalg::matrix::{Matrix, Vector};

/// Maximum matrix dimension for Cholesky (compile-time bound)
pub const MAX_CHOL_DIM: usize = 64;

/// Result of Cholesky decomposition
#[derive(Debug)]
#[repr(C)]
pub struct CholeskyResult<const N: usize> {
    /// Lower triangular matrix L where A = L * L^T
    pub lower: Matrix<N, N>,
    /// Whether decomposition succeeded
    pub success: bool,
    /// Error code if failed (0 = positive definite, >0 = index where failed)
    pub error_index: usize,
}

impl<const N: usize> CholeskyResult<N> {
    pub const fn empty() -> Self {
        Self {
            lower: Matrix::<N, N>::zeros(),
            success: false,
            error_index: 0,
        }
    }
}

/// Perform Cholesky decomposition on a symmetric positive-definite matrix
/// Returns lower triangular matrix L such that A = L * L^T
/// 
/// Algorithm: In-place Cholesky-Banachiewicz
/// Complexity: O(N^3/3) floating point operations
#[inline]
pub fn cholesky<const N: usize>(matrix: &Matrix<N, N>) -> CholeskyResult<N> {
    let mut result = CholeskyResult::<N> {
        lower: Matrix::<N, N>::zeros(),
        success: true,
        error_index: 0,
    };
    
    // Copy input to working matrix
    // We work on a copy to preserve the original
    let mut l = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            if let Some(val) = matrix.get(i, j) {
                l[i][j] = val;
            }
        }
    }
    
    // Cholesky-Banachiewicz algorithm
    for i in 0..N {
        for j in 0..=i {
            let mut sum = 0.0;
            
            if j == i {
                // Diagonal element
                for k in 0..j {
                    sum += l[j][k] * l[j][k];
                }
                
                let diag = l[j][j] - sum;
                if diag <= 0.0 {
                    // Matrix is not positive definite
                    result.success = false;
                    result.error_index = i;
                    return result;
                }
                l[j][j] = diag.sqrt();
            } else {
                // Off-diagonal element
                for k in 0..j {
                    sum += l[i][k] * l[j][k];
                }
                
                if l[j][j].abs() < 1e-15 {
                    result.success = false;
                    result.error_index = i;
                    return result;
                }
                l[i][j] = (l[i][j] - sum) / l[j][j];
            }
        }
        
        // Zero out upper triangle
        for j in (i+1)..N {
            l[i][j] = 0.0;
        }
    }
    
    // Copy result to matrix
    for i in 0..N {
        for j in 0..N {
            result.lower.set(i, j, l[i][j]);
        }
    }
    
    result
}

/// Solve linear system Ax = b using Cholesky decomposition
/// where A is symmetric positive-definite
/// 
/// Steps:
/// 1. Compute L where A = L * L^T
/// 2. Solve Ly = b (forward substitution)
/// 3. Solve L^T x = y (backward substitution)
#[inline]
pub fn solve_cholesky<const N: usize>(
    matrix: &Matrix<N, N>,
    b: &Vector<N>,
) -> Option<Vector<N>> {
    let chol = cholesky(matrix);
    if !chol.success {
        return None;
    }
    
    let l = chol.lower;
    let mut y = [0.0; N];
    let mut x = [0.0; N];
    
    // Forward substitution: Ly = b
    for i in 0..N {
        let mut sum = 0.0;
        for j in 0..i {
            sum += unsafe { l.get_unchecked(i, j) } * y[j];
        }
        let lii = unsafe { l.get_unchecked(i, i) };
        if lii.abs() < 1e-15 {
            return None;
        }
        y[i] = (b.data[i][0] - sum) / lii;
    }
    
    // Backward substitution: L^T x = y
    for i in (0..N).rev() {
        let mut sum = 0.0;
        for j in (i+1)..N {
            sum += unsafe { l.get_unchecked(j, i) } * x[j]; // L^T[i][j] = L[j][i]
        }
        let lii = unsafe { l.get_unchecked(i, i) };
        if lii.abs() < 1e-15 {
            return None;
        }
        x[i] = (y[i] - sum) / lii;
    }
    
    Some(Vector::from_array(x))
}

/// Generate correlated random samples using Cholesky decomposition
/// Given uncorrelated standard normal samples z, produces correlated samples x = L * z
/// where L is the Cholesky factor of the covariance matrix
#[inline]
pub fn generate_correlated_samples<const N: usize>(
    cholesky_factor: &Matrix<N, N>,
    uncorrelated: &[f64; N],
) -> [f64; N] {
    let mut result = [0.0; N];
    
    for i in 0..N {
        let mut sum = 0.0;
        for j in 0..=i {
            sum += unsafe { cholesky_factor.get_unchecked(i, j) } * uncorrelated[j];
        }
        result[i] = sum;
    }
    
    result
}

/// Compute log determinant of a positive-definite matrix via Cholesky
/// log(det(A)) = 2 * sum(log(diag(L)))
#[inline]
pub fn log_determinant_cholesky<const N: usize>(matrix: &Matrix<N, N>) -> Option<f64> {
    let chol = cholesky(matrix);
    if !chol.success {
        return None;
    }
    
    let mut log_det = 0.0;
    for i in 0..N {
        let diag = chol.lower.data[i][i];
        if diag <= 0.0 {
            return None;
        }
        log_det += diag.ln();
    }
    
    Some(2.0 * log_det)
}

/// Compute matrix inverse using Cholesky decomposition
/// For symmetric positive-definite matrices only
#[inline]
pub fn inverse_cholesky<const N: usize>(matrix: &Matrix<N, N>) -> Option<Matrix<N, N>> {
    let chol = cholesky(matrix);
    if !chol.success {
        return None;
    }
    
    let l = chol.lower;
    let mut inv = Matrix::<N, N>::zeros();
    
    // Invert L using forward substitution column by column
    let mut l_inv = [[0.0; N]; N];
    
    for col in 0..N {
        for row in col..N {
            let mut sum = 0.0;
            for k in col..row {
                sum += unsafe { l.get_unchecked(row, k) } * l_inv[k][col];
            }
            
            if row == col {
                let diag = unsafe { l.get_unchecked(row, row) };
                if diag.abs() < 1e-15 {
                    return None;
                }
                l_inv[row][col] = 1.0 / diag;
            } else {
                let diag = unsafe { l.get_unchecked(row, row) };
                if diag.abs() < 1e-15 {
                    return None;
                }
                l_inv[row][col] = -sum / diag;
            }
        }
    }
    
    // Compute A^-1 = (L^-1)^T * L^-1
    for i in 0..N {
        for j in i..N {
            let mut sum = 0.0;
            for k in 0..N {
                sum += l_inv[k][i] * l_inv[k][j];
            }
            inv.set(i, j, sum);
            inv.set(j, i, sum); // Symmetric
        }
    }
    
    Some(inv)
}

/// Verify Cholesky decomposition by computing L * L^T and comparing to original
#[inline]
pub fn verify_cholesky<const N: usize>(
    original: &Matrix<N, N>,
    result: &CholeskyResult<N>,
    epsilon: f64,
) -> bool {
    if !result.success {
        return false;
    }
    
    let l = &result.lower;
    let lt = l.transpose();
    
    for i in 0..N {
        for j in 0..N {
            let mut computed = 0.0;
            for k in 0..N {
                computed += unsafe { l.get_unchecked(i, k) } * unsafe { lt.get_unchecked(k, j) };
            }
            
            if let Some(orig) = original.get(i, j) {
                if (computed - orig).abs() > epsilon {
                    return false;
                }
            }
        }
    }
    
    true
}

/// Near-positive-definite correction: add small value to diagonal until PD
#[inline]
pub fn make_positive_definite<const N: usize>(
    matrix: &Matrix<N, N>,
    max_iterations: usize,
    initial_epsilon: f64,
) -> Option<Matrix<N, N>> {
    let mut corrected = *matrix;
    let mut epsilon = initial_epsilon;
    
    for _ in 0..max_iterations {
        let result = cholesky(&corrected);
        if result.success {
            return Some(corrected);
        }
        
        // Add epsilon to diagonal
        for i in 0..N {
            if let Some(val) = corrected.get(i, i) {
                corrected.set(i, i, val + epsilon);
            }
        }
        
        epsilon *= 2.0;
    }
    
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cholesky_basic() {
        // Simple 3x3 positive definite matrix
        let a = Matrix::<3, 3>::from_array([
            [4.0, 12.0, -16.0],
            [12.0, 37.0, -43.0],
            [-16.0, -43.0, 98.0],
        ]);
        
        let result = cholesky(&a);
        assert!(result.success);
        
        // Verify: L should be lower triangular
        assert!((result.lower.get(0, 1).unwrap() - 0.0).abs() < 1e-10);
        assert!((result.lower.get(0, 2).unwrap() - 0.0).abs() < 1e-10);
        assert!((result.lower.get(1, 2).unwrap() - 0.0).abs() < 1e-10);
        
        // Verify reconstruction
        assert!(verify_cholesky(&a, &result, 1e-10));
    }
    
    #[test]
    fn test_solve_cholesky() {
        let a = Matrix::<3, 3>::from_array([
            [4.0, 12.0, -16.0],
            [12.0, 37.0, -43.0],
            [-16.0, -43.0, 98.0],
        ]);
        
        let b = Vector::from_array([1.0, 2.0, 3.0]);
        
        let x = solve_cholesky(&a, &b);
        assert!(x.is_some());
        
        // Verify: Ax should equal b
        let x = x.unwrap();
        let ax = a.matmul(&x.transpose());
        
        for i in 0..3 {
            let computed = ax.get(i, 0).unwrap();
            let expected = b.get(i).unwrap();
            assert!((computed - expected).abs() < 1e-10);
        }
    }
    
    #[test]
    fn test_log_determinant() {
        let a = Matrix::<2, 2>::from_array([
            [4.0, 1.0],
            [1.0, 3.0],
        ]);
        
        let log_det = log_determinant_cholesky(&a);
        assert!(log_det.is_some());
        
        // det(A) = 4*3 - 1*1 = 11, log(11) ≈ 2.398
        let expected = 11.0_f64.ln();
        assert!((log_det.unwrap() - expected).abs() < 1e-10);
    }
    
    #[test]
    fn test_non_positive_definite() {
        // This matrix is not positive definite
        let a = Matrix::<2, 2>::from_array([
            [1.0, 2.0],
            [2.0, 1.0],
        ]);
        
        let result = cholesky(&a);
        assert!(!result.success);
    }
    
    #[test]
    fn test_generate_correlated_samples() {
        // Covariance matrix for 2 correlated variables
        let cov = Matrix::<2, 2>::from_array([
            [1.0, 0.8],
            [0.8, 1.0],
        ]);
        
        let chol = cholesky(&cov);
        assert!(chol.success);
        
        // Uncorrelated standard normals
        let z = [1.0, 0.5];
        
        let correlated = generate_correlated_samples(&chol.lower, &z);
        
        // First element should just be z[0] (since L[0][0] = 1)
        assert!((correlated[0] - 1.0).abs() < 1e-10);
    }
}
