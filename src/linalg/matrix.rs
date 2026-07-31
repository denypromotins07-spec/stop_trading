//! High-Speed Linear Algebra - Dense Matrix
//! 
//! Implements cache-line aligned, fixed-size dense matrix structs for covariance
//! and correlation calculations. Uses flat, contiguous memory layouts to maximize
//! CPU cache hits during portfolio variance computations.

use core::ops::{Add, Sub, Mul};
use core::marker::PhantomData;

/// Maximum matrix dimension supported (compile-time constant)
pub const MAX_MATRIX_DIM: usize = 64;

/// Cache-line padded matrix for N×M dimensions using const generics
/// Memory layout is row-major for cache-friendly access patterns
#[derive(Debug, Clone)]
#[repr(C, align(64))]
pub struct Matrix<const N: usize, const M: usize> {
    /// Flat array storage in row-major order
    data: [[f64; M]; N],
}

impl<const N: usize, const M: usize> Matrix<N, M> {
    /// Create a new zero-initialized matrix
    #[inline]
    pub const fn zeros() -> Self {
        Self {
            data: [[0.0; M]; N],
        }
    }
    
    /// Create an identity matrix (only valid for square matrices)
    #[inline]
    pub fn identity() -> Self {
        assert!(N == M, "Identity matrix requires N == M");
        let mut m = Self::zeros();
        for i in 0..N {
            m.data[i][i] = 1.0;
        }
        m
    }
    
    /// Create matrix from flat array
    #[inline]
    pub fn from_array(data: [[f64; M]; N]) -> Self {
        Self { data }
    }
    
    /// Get element at (row, col) with bounds checking
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        if row < N && col < M {
            Some(self.data[row][col])
        } else {
            None
        }
    }
    
    /// Get element at (row, col) without bounds checking (unsafe but fast)
    #[inline]
    pub unsafe fn get_unchecked(&self, row: usize, col: usize) -> f64 {
        *self.data.get_unchecked(row).get_unchecked(col)
    }
    
    /// Set element at (row, col) with bounds checking
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: f64) -> bool {
        if row < N && col < M {
            self.data[row][col] = value;
            true
        } else {
            false
        }
    }
    
    /// Set element without bounds checking
    #[inline]
    pub unsafe fn set_unchecked(&mut self, row: usize, col: usize, value: f64) {
        *self.data.get_unchecked_mut(row).get_unchecked_mut(col) = value;
    }
    
    /// Get row slice
    #[inline]
    pub fn row(&self, row: usize) -> Option<&[f64]> {
        if row < N {
            Some(&self.data[row])
        } else {
            None
        }
    }
    
    /// Get mutable row slice
    #[inline]
    pub fn row_mut(&mut self, row: usize) -> Option<&mut [f64]> {
        if row < N {
            Some(&mut self.data[row])
        } else {
            None
        }
    }
    
    /// Transpose the matrix (M×N result)
    #[inline]
    pub fn transpose(&self) -> Matrix<M, N> {
        let mut result = Matrix::<M, N>::zeros();
        for i in 0..N {
            for j in 0..M {
                result.data[j][i] = self.data[i][j];
            }
        }
        result
    }
    
    /// Matrix addition
    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        let mut result = Self::zeros();
        for i in 0..N {
            for j in 0..M {
                result.data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }
        result
    }
    
    /// Matrix subtraction
    #[inline]
    pub fn sub(&self, other: &Self) -> Self {
        let mut result = Self::zeros();
        for i in 0..N {
            for j in 0..M {
                result.data[i][j] = self.data[i][j] - other.data[i][j];
            }
        }
        result
    }
    
    /// Matrix multiplication with K×N matrix (result is K×M)
    #[inline]
    pub fn matmul<const K: usize>(&self, other: &Matrix<K, N>) -> Matrix<K, M> {
        let mut result = Matrix::<K, M>::zeros();
        for i in 0..K {
            for j in 0..M {
                let mut sum = 0.0;
                for k in 0..N {
                    sum += other.data[i][k] * self.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        result
    }
    
    /// Multiply by scalar
    #[inline]
    pub fn scale(&self, scalar: f64) -> Self {
        let mut result = Self::zeros();
        for i in 0..N {
            for j in 0..M {
                result.data[i][j] = self.data[i][j] * scalar;
            }
        }
        result
    }
    
    /// Element-wise multiplication (Hadamard product)
    #[inline]
    pub fn hadamard(&self, other: &Self) -> Self {
        let mut result = Self::zeros();
        for i in 0..N {
            for j in 0..M {
                result.data[i][j] = self.data[i][j] * other.data[i][j];
            }
        }
        result
    }
    
    /// Calculate Frobenius norm
    #[inline]
    pub fn frobenius_norm(&self) -> f64 {
        let mut sum = 0.0;
        for i in 0..N {
            for j in 0..M {
                sum += self.data[i][j] * self.data[i][j];
            }
        }
        sum.sqrt()
    }
    
    /// Calculate sum of all elements
    #[inline]
    pub fn sum(&self) -> f64 {
        let mut total = 0.0;
        for i in 0..N {
            for j in 0..M {
                total += self.data[i][j];
            }
        }
        total
    }
    
    /// Calculate mean of all elements
    #[inline]
    pub fn mean(&self) -> f64 {
        self.sum() / (N * M) as f64
    }
    
    /// Get raw pointer to data for FFI/SIMD operations
    #[inline]
    pub fn as_ptr(&self) -> *const f64 {
        self.data.as_ptr() as *const f64
    }
    
    /// Get mutable raw pointer to data
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut f64 {
        self.data.as_mut_ptr() as *mut f64
    }
    
    /// Fill matrix with value
    #[inline]
    pub fn fill(&mut self, value: f64) {
        for i in 0..N {
            for j in 0..M {
                self.data[i][j] = value;
            }
        }
    }
    
    /// Check if matrix is symmetric (for square matrices)
    #[inline]
    pub fn is_symmetric(&self, epsilon: f64) -> bool {
        if N != M {
            return false;
        }
        for i in 0..N {
            for j in (i+1)..M {
                if (self.data[i][j] - self.data[j][i]).abs() > epsilon {
                    return false;
                }
            }
        }
        true
    }
}

impl<const N: usize, const M: usize> Default for Matrix<N, M> {
    fn default() -> Self {
        Self::zeros()
    }
}

impl<const N: usize, const M: usize> Add for Matrix<N, M> {
    type Output = Self;
    
    fn add(self, other: Self) -> Self::Output {
        self.add(&other)
    }
}

impl<const N: usize, const M: usize> Sub for Matrix<N, M> {
    type Output = Self;
    
    fn sub(self, other: Self) -> Self::Output {
        self.sub(&other)
    }
}

/// Specialized 8×8 matrix for small covariance calculations
pub type Matrix8x8 = Matrix<8, 8>;

/// Specialized 16×16 matrix for medium portfolio covariance
pub type Matrix16x16 = Matrix<16, 16>;

/// Specialized 32×32 matrix for large portfolio covariance
pub type Matrix32x32 = Matrix<32, 32>;

/// Column vector type
pub type Vector<const N: usize> = Matrix<N, 1>;

impl<const N: usize> Vector<N> {
    /// Create vector from array
    #[inline]
    pub fn from_array(values: [f64; N]) -> Self {
        let mut v = Self::zeros();
        for i in 0..N {
            v.data[i][0] = values[i];
        }
        v
    }
    
    /// Get element
    #[inline]
    pub fn get(&self, idx: usize) -> Option<f64> {
        if idx < N {
            Some(self.data[idx][0])
        } else {
            None
        }
    }
    
    /// Dot product with another vector
    #[inline]
    pub fn dot(&self, other: &Self) -> f64 {
        let mut sum = 0.0;
        for i in 0..N {
            sum += self.data[i][0] * other.data[i][0];
        }
        sum
    }
    
    /// L2 norm
    #[inline]
    pub fn norm(&self) -> f64 {
        self.dot(self).sqrt()
    }
    
    /// Normalize vector
    #[inline]
    pub fn normalize(&self) -> Self {
        let n = self.norm();
        if n > 1e-15 {
            self.scale(1.0 / n)
        } else {
            Self::zeros()
        }
    }
}

/// Compute outer product of two vectors (result is N×M matrix)
#[inline]
pub fn outer_product<const N: usize, const M: usize>(
    a: &Vector<N>,
    b: &Vector<M>,
) -> Matrix<N, M> {
    let mut result = Matrix::<N, M>::zeros();
    for i in 0..N {
        for j in 0..M {
            result.data[i][j] = a.data[i][0] * b.data[j][0];
        }
    }
    result
}

/// Compute covariance matrix from data samples
/// Data is organized as rows (samples) × columns (features)
#[inline]
pub fn compute_covariance<const FEATURES: usize, const SAMPLES: usize>(
    data: &Matrix<SAMPLES, FEATURES>,
) -> Matrix<FEATURES, FEATURES> {
    // First compute means
    let mut means = [0.0; FEATURES];
    for j in 0..FEATURES {
        let mut sum = 0.0;
        for i in 0..SAMPLES {
            sum += unsafe { data.get_unchecked(i, j) };
        }
        means[j] = sum / SAMPLES as f64;
    }
    
    // Center the data and compute covariance
    let mut covar = Matrix::<FEATURES, FEATURES>::zeros();
    let denom = (SAMPLES - 1) as f64;
    
    for i in 0..FEATURES {
        for j in i..FEATURES {
            let mut sum = 0.0;
            for k in 0..SAMPLES {
                let di = unsafe { data.get_unchecked(k, i) } - means[i];
                let dj = unsafe { data.get_unchecked(k, j) } - means[j];
                sum += di * dj;
            }
            let cov = sum / denom;
            covar.data[i][j] = cov;
            covar.data[j][i] = cov; // Symmetric
        }
    }
    
    covar
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_matrix_creation() {
        let m: Matrix<3, 3> = Matrix::zeros();
        assert_eq!(m.get(0, 0), Some(0.0));
        
        let id = Matrix::<4, 4>::identity();
        assert_eq!(id.get(0, 0), Some(1.0));
        assert_eq!(id.get(1, 1), Some(1.0));
        assert_eq!(id.get(0, 1), Some(0.0));
    }
    
    #[test]
    fn test_matrix_transpose() {
        let mut m = Matrix::<2, 3>::zeros();
        m.set(0, 0, 1.0);
        m.set(0, 1, 2.0);
        m.set(0, 2, 3.0);
        m.set(1, 0, 4.0);
        m.set(1, 1, 5.0);
        m.set(1, 2, 6.0);
        
        let t = m.transpose();
        assert_eq!(t.get(0, 0), Some(1.0));
        assert_eq!(t.get(1, 0), Some(2.0));
        assert_eq!(t.get(2, 0), Some(3.0));
        assert_eq!(t.get(0, 1), Some(4.0));
    }
    
    #[test]
    fn test_matrix_multiply() {
        let a = Matrix::<2, 3>::from_array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]);
        let b = Matrix::<3, 2>::from_array([
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
        ]);
        
        let c = a.matmul(&b);
        assert_eq!(c.get(0, 0), Some(58.0)); // 1*7 + 2*9 + 3*11
        assert_eq!(c.get(0, 1), Some(64.0)); // 1*8 + 2*10 + 3*12
        assert_eq!(c.get(1, 0), Some(139.0)); // 4*7 + 5*9 + 6*11
        assert_eq!(c.get(1, 1), Some(154.0));
    }
    
    #[test]
    fn test_vector_operations() {
        let v1 = Vector::<4>::from_array([1.0, 2.0, 3.0, 4.0]);
        let v2 = Vector::<4>::from_array([5.0, 6.0, 7.0, 8.0]);
        
        let dot = v1.dot(&v2);
        assert_eq!(dot, 70.0); // 1*5 + 2*6 + 3*7 + 4*8
        
        let norm = v1.norm();
        assert!((norm - 5.477225575051661).abs() < 1e-10);
    }
    
    #[test]
    fn test_covariance() {
        // Simple 3-sample, 2-feature dataset
        let data = Matrix::<3, 2>::from_array([
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
        ]);
        
        let cov = compute_covariance(&data);
        // Features are perfectly correlated (y = 2x)
        assert!(cov.is_symmetric(1e-10));
    }
}
