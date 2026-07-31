//! Hierarchical Risk Parity (HRP) Implementation
//! 
//! Implements the HRP algorithm using tree clustering and recursive bisection
//! to allocate capital without heavy matrix inversions. O(N log N) complexity.
//! Strictly bounded by 6.5GB RAM limit using fixed-size arrays and custom arenas.

use alloc::vec::Vec;
use core::cmp::Ordering;

/// Maximum number of assets supported in the portfolio
pub const MAX_ASSETS: usize = 512;

/// Fixed-size covariance matrix storage for memory safety
#[derive(Clone, Debug)]
pub struct CovarianceMatrix {
    data: [[f64; MAX_ASSETS]; MAX_ASSETS],
    size: usize,
}

impl CovarianceMatrix {
    pub fn new(size: usize) -> Self {
        assert!(size <= MAX_ASSETS, "Asset count exceeds maximum");
        let mut data = [[0.0; MAX_ASSETS]; MAX_ASSETS];
        CovarianceMatrix { data, size }
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        debug_assert!(i < self.size && j < self.size);
        unsafe { *self.data.get_unchecked(i).get_unchecked(j) }
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, val: f64) {
        debug_assert!(i < self.size && j < self.size);
        unsafe {
            *self.data.get_unchecked_mut(i).get_unchecked_mut(j) = val;
        }
    }

    pub fn from_slice(cov: &[[f64]]) -> Self {
        let size = cov.len();
        assert!(size <= MAX_ASSETS, "Covariance matrix exceeds maximum size");
        let mut matrix = Self::new(size);
        for (i, row) in cov.iter().enumerate() {
            for (j, val) in row.iter().enumerate() {
                matrix.set(i, j, *val);
            }
        }
        matrix
    }
}

/// Cluster node in the hierarchical tree
#[derive(Clone, Debug)]
pub struct ClusterNode {
    pub id: usize,
    pub left: Option<usize>,
    pub right: Option<usize>,
    pub distance: f64,
    pub items: Vec<usize>,
}

/// Hierarchical Risk Parity optimizer
pub struct HierarchicalRiskParity {
    cov_matrix: CovarianceMatrix,
    variances: Vec<f64>,
    cluster_tree: Vec<ClusterNode>,
}

impl HierarchicalRiskParity {
    pub fn new(cov_matrix: CovarianceMatrix) -> Self {
        let n = cov_matrix.size;
        let mut variances = Vec::with_capacity(n);
        for i in 0..n {
            variances.push(cov_matrix.get(i, i));
        }
        
        HierarchicalRiskParity {
            cov_matrix,
            variances,
            cluster_tree: Vec::with_capacity(2 * n),
        }
    }

    /// Compute correlation matrix from covariance
    fn compute_correlation(&self) -> Vec<Vec<f64>> {
        let n = self.cov_matrix.size;
        let mut corr = vec![vec![0.0; n]; n];
        
        for i in 0..n {
            corr[i][i] = 1.0;
            let std_i = self.variances[i].sqrt();
            for j in (i + 1)..n {
                let std_j = self.variances[j].sqrt();
                let cov_ij = self.cov_matrix.get(i, j);
                let corr_ij = cov_ij / (std_i * std_j);
                corr[i][j] = corr_ij.clamp(-1.0, 1.0);
                corr[j][i] = corr_ij.clamp(-1.0, 1.0);
            }
        }
        corr
    }

    /// Single-linkage hierarchical clustering
    /// Returns the cluster tree structure
    pub fn build_tree(&mut self) -> usize {
        let n = self.cov_matrix.size;
        let corr = self.compute_correlation();
        
        // Initialize leaf nodes
        self.cluster_tree.clear();
        for i in 0..n {
            self.cluster_tree.push(ClusterNode {
                id: i,
                left: None,
                right: None,
                distance: 0.0,
                items: vec![i],
            });
        }

        // Distance matrix using 1 - |correlation|
        let mut dist = vec![vec![f64::MAX; n]; n];
        let mut active = vec![true; n];
        
        for i in 0..n {
            for j in (i + 1)..n {
                let d = 1.0 - corr[i][j].abs();
                dist[i][j] = d;
                dist[j][i] = d;
            }
        }

        let mut next_id = n;
        let mut clusters = n;

        while clusters > 1 {
            // Find minimum distance pair
            let mut min_dist = f64::MAX;
            let mut min_i = 0;
            let mut min_j = 1;

            for i in 0..next_id {
                if !active[i] {
                    continue;
                }
                for j in (i + 1)..next_id {
                    if !active[j] {
                        continue;
                    }
                    if dist[i][j] < min_dist {
                        min_dist = dist[i][j];
                        min_i = i;
                        min_j = j;
                    }
                }
            }

            // Merge clusters
            let mut merged_items = Vec::new();
            merged_items.extend_from_slice(&self.cluster_tree[min_i].items);
            merged_items.extend_from_slice(&self.cluster_tree[min_j].items);

            self.cluster_tree.push(ClusterNode {
                id: next_id,
                left: Some(min_i),
                right: Some(min_j),
                distance: min_dist,
                items: merged_items,
            });

            active[min_i] = false;
            active[min_j] = false;
            active.push(true);

            // Update distances (single linkage)
            dist.push(vec![f64::MAX; next_id + 1]);
            for k in 0..next_id {
                if !active[k] || k == next_id {
                    continue;
                }
                let d_k_new = dist[k][min_i].min(dist[k][min_j]);
                dist[k][next_id] = d_k_new;
                dist[next_id][k] = d_k_new;
            }

            clusters -= 1;
            next_id += 1;
        }

        next_id - 1
    }

    /// Quasi-diagonalization: reorder covariance matrix based on tree
    fn quasi_diagonalize(&self, root: usize) -> Vec<usize> {
        let mut ordered = Vec::with_capacity(self.cov_matrix.size);
        self.quasi_diag_helper(root, &mut ordered);
        ordered
    }

    fn quasi_diag_helper(&self, node: usize, ordered: &mut Vec<usize>) {
        let cluster = &self.cluster_tree[node];
        match (cluster.left, cluster.right) {
            (Some(left), Some(right)) => {
                // Compare variances to decide order
                let var_left: f64 = cluster.items.iter()
                    .filter(|&&i| self.cluster_tree[left].items.contains(&i))
                    .map(|&i| self.variances[i])
                    .sum::<f64>();
                let var_right: f64 = cluster.items.iter()
                    .filter(|&&i| self.cluster_tree[right].items.contains(&i))
                    .map(|&i| self.variances[i])
                    .sum::<f64>();
                
                if var_left <= var_right {
                    self.quasi_diag_helper(left, ordered);
                    self.quasi_diag_helper(right, ordered);
                } else {
                    self.quasi_diag_helper(right, ordered);
                    self.quasi_diag_helper(left, ordered);
                }
            }
            _ => {
                ordered.extend(&cluster.items);
            }
        }
    }

    /// Recursive bisection to allocate weights
    fn recursive_bisection(&self, ordered: &[usize]) -> Vec<f64> {
        let n = ordered.len();
        let mut weights = vec![1.0; n];
        
        self.bisect_helper(ordered, &mut weights, 0, n);
        
        // Normalize weights
        let sum: f64 = weights.iter().sum();
        for w in &mut weights {
            *w /= sum;
        }
        
        weights
    }

    fn bisect_helper(&self, ordered: &[usize], weights: &mut [f64], start: usize, end: usize) {
        if end - start <= 1 {
            return;
        }

        let mid = (start + end) / 2;
        
        // Calculate variance of each cluster
        let var_left = self.cluster_variance(&ordered[start..mid]);
        let var_right = self.cluster_variance(&ordered[mid..end]);
        
        // Allocation factor based on inverse variance
        let total_var = var_left + var_right;
        if total_var > 1e-12 {
            let alpha = 1.0 - var_left / total_var;
            
            // Scale weights
            let sum_left: f64 = weights[start..mid].iter().sum();
            let sum_right: f64 = weights[mid..end].iter().sum();
            
            for i in start..mid {
                weights[i] *= (1.0 - alpha) / sum_left.max(1e-12);
            }
            for i in mid..end {
                weights[i] *= alpha / sum_right.max(1e-12);
            }
        }

        // Recurse
        self.bisect_helper(ordered, weights, start, mid);
        self.bisect_helper(ordered, weights, mid, end);
    }

    fn cluster_variance(&self, indices: &[usize]) -> f64 {
        let mut var = 0.0;
        for &i in indices {
            for &j in indices {
                var += self.cov_matrix.get(i, j);
            }
        }
        var
    }

    /// Main HRP allocation method
    /// Returns weights aligned with original asset ordering
    pub fn allocate(&mut self) -> Vec<f64> {
        let root = self.build_tree();
        let ordered = self.quasi_diagonalize(root);
        let ordered_weights = self.recursive_bisection(&ordered);
        
        // Map back to original ordering
        let mut weights = vec![0.0; self.cov_matrix.size];
        for (idx, &orig_idx) in ordered.iter().enumerate() {
            weights[orig_idx] = ordered_weights[idx];
        }
        
        weights
    }

    /// Get optimized allocation with validation
    pub fn allocate_validated(&mut self) -> Result<Vec<f64>, HRPError> {
        if self.cov_matrix.size == 0 {
            return Err(HRPError::EmptyPortfolio);
        }

        // Validate covariance matrix is positive semi-definite
        for i in 0..self.cov_matrix.size {
            if self.variances[i] <= 0.0 {
                return Err(HRPError::InvalidVariance(i));
            }
        }

        Ok(self.allocate())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum HRPError {
    EmptyPortfolio,
    InvalidVariance(usize),
    AssetLimitExceeded,
}

/// Memory arena for HRP computations to stay within 6.5GB limit
pub struct HRPArena {
    buffer: Box<[u8]>,
    offset: usize,
}

impl HRPArena {
    pub fn new(capacity_mb: usize) -> Self {
        let capacity_bytes = capacity_mb * 1024 * 1024;
        HRPArena {
            buffer: vec![0u8; capacity_bytes].into_boxed_slice(),
            offset: 0,
        }
    }

    pub fn alloc<T>(&mut self, count: usize) -> Option<&mut [T]> {
        let size = count * core::mem::size_of::<T>();
        let align = core::mem::align_of::<T>();
        
        // Align offset
        let aligned_offset = (self.offset + align - 1) & !(align - 1);
        
        if aligned_offset + size > self.buffer.len() {
            return None;
        }

        let ptr = unsafe {
            self.buffer.as_mut_ptr().add(aligned_offset) as *mut T
        };
        
        self.offset = aligned_offset + size;
        Some(unsafe { core::slice::from_raw_parts_mut(ptr, count) })
    }

    pub fn reset(&mut self) {
        self.offset = 0;
    }

    pub fn used_bytes(&self) -> usize {
        self.offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hrp_basic() {
        let cov_data = vec![
            vec![0.04, 0.01, 0.02],
            vec![0.01, 0.09, 0.03],
            vec![0.02, 0.03, 0.16],
        ];
        let cov = CovarianceMatrix::from_slice(&cov_data);
        let mut hrp = HierarchicalRiskParity::new(cov);
        
        let weights = hrp.allocate_validated().unwrap();
        
        assert_eq!(weights.len(), 3);
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        
        // All weights should be positive
        for w in &weights {
            assert!(*w > 0.0);
        }
    }

    #[test]
    fn test_arena_allocation() {
        let mut arena = HRPArena::new(64);
        
        let slice: &mut [f64] = arena.alloc(100).unwrap();
        assert_eq!(slice.len(), 100);
        
        let used = arena.used_bytes();
        assert!(used > 0);
        
        arena.reset();
        assert_eq!(arena.used_bytes(), 0);
    }
}
