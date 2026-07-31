//! Lightweight Isolation Forest for Anomaly Detection
//! 
//! Implements a fixed-depth Isolation Forest in pure Rust to detect market microstructure anomalies.
//! Uses pre-allocated tree nodes and cache-line padded structs for memory efficiency.

use std::sync::atomic::{AtomicU64, Ordering};

/// Cache-line padded struct to prevent false sharing
#[repr(align(64))]
#[derive(Debug, Clone)]
pub struct PaddedNode {
    pub split_feature: u32,
    pub split_value: f64,
    pub left_child: i32,
    pub right_child: i32,
    pub size: u32,
    pub is_leaf: bool,
    pub depth: u32,
    _padding: [u8; 16], // Padding to 64 bytes
}

impl Default for PaddedNode {
    fn default() -> Self {
        PaddedNode {
            split_feature: 0,
            split_value: 0.0,
            left_child: -1,
            right_child: -1,
            size: 0,
            is_leaf: true,
            depth: 0,
            _padding: [0; 16],
        }
    }
}

/// Pre-allocated node pool for zero-allocation tree building
pub struct NodePool<const MAX_NODES: usize> {
    nodes: [PaddedNode; MAX_NODES],
    used: AtomicU64,
}

impl<const MAX_NODES: usize> NodePool<MAX_NODES> {
    pub const fn new() -> Self {
        NodePool {
            nodes: [const { PaddedNode::default() }; MAX_NODES],
            used: AtomicU64::new(0),
        }
    }

    pub fn allocate(&self) -> Option<usize> {
        let idx = self.used.fetch_add(1, Ordering::Relaxed) as usize;
        if idx < MAX_NODES {
            Some(idx)
        } else {
            None
        }
    }

    pub fn reset(&self) {
        self.used.store(0, Ordering::Relaxed);
    }

    pub fn get(&self, idx: usize) -> &PaddedNode {
        &self.nodes[idx]
    }

    pub fn get_mut(&mut self, idx: usize) -> &mut PaddedNode {
        &mut self.nodes[idx]
    }

    pub fn used_count(&self) -> usize {
        self.used.load(Ordering::Relaxed) as usize
    }
}

/// Single isolation tree with pre-allocated nodes
pub struct IsolationTree<const MAX_NODES: usize> {
    pub root: i32,
    pub max_depth: u32,
    pub node_pool: NodePool<MAX_NODES>,
}

impl<const MAX_NODES: usize> IsolationTree<MAX_NODES> {
    pub fn new(max_depth: u32) -> Self {
        IsolationTree {
            root: -1,
            max_depth,
            node_pool: NodePool::new(),
        }
    }

    /// Build tree from sample data (simplified implementation)
    pub fn build(&mut self, data: &[Vec<f64>], rng_seed: u64) {
        self.node_pool.reset();
        
        if data.is_empty() {
            return;
        }

        let n_features = data[0].len();
        if n_features == 0 {
            return;
        }

        // Create root node
        if let Some(root_idx) = self.node_pool.allocate() {
            self.root = root_idx as i32;
            
            let node = self.node_pool.get_mut(root_idx);
            node.size = data.len() as u32;
            node.depth = 0;
            node.is_leaf = false;
            
            // Recursively build tree
            self.build_recursive(data, root_idx, 0, rng_seed);
        }
    }

    fn build_recursive(&mut self, data: &[Vec<f64>], node_idx: usize, current_depth: u32, rng_seed: u64) {
        if current_depth >= self.max_depth || data.len() <= 1 {
            let node = self.node_pool.get_mut(node_idx);
            node.is_leaf = true;
            node.depth = current_depth;
            return;
        }

        // Simple random split (production would use proper RNG)
        let n_features = data[0].len();
        let feature_idx = ((rng_seed + current_depth as u64) as usize) % n_features;
        
        // Find min/max for this feature
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;
        for row in data {
            min_val = min_val.min(row[feature_idx]);
            max_val = max_val.max(row[feature_idx]);
        }

        if min_val >= max_val {
            let node = self.node_pool.get_mut(node_idx);
            node.is_leaf = true;
            node.depth = current_depth;
            return;
        }

        // Random split value
        let split_value = min_val + (max_val - min_val) * 0.5;

        // Split data
        let mut left_data = Vec::with_capacity(data.len());
        let mut right_data = Vec::with_capacity(data.len());
        
        for row in data {
            if row[feature_idx] < split_value {
                left_data.push(row.clone());
            } else {
                right_data.push(row.clone());
            }
        }

        if left_data.is_empty() || right_data.is_empty() {
            let node = self.node_pool.get_mut(node_idx);
            node.is_leaf = true;
            node.depth = current_depth;
            return;
        }

        // Create child nodes
        if let Some(left_idx) = self.node_pool.allocate() {
            let node = self.node_pool.get_mut(node_idx);
            node.split_feature = feature_idx as u32;
            node.split_value = split_value;
            node.left_child = left_idx as i32;
            node.size = left_data.len() as u32;

            self.build_recursive(&left_data, left_idx, current_depth + 1, rng_seed);
        }

        if let Some(right_idx) = self.node_pool.allocate() {
            let node = self.node_pool.get_mut(node_idx);
            node.split_feature = feature_idx as u32;
            node.split_value = split_value;
            node.right_child = right_idx as i32;
            node.size = right_data.len() as u32;

            self.build_recursive(&right_data, right_idx, current_depth + 1, rng_seed);
        }
    }

    /// Calculate path length for a single sample
    pub fn path_length(&self, sample: &[f64]) -> f64 {
        if self.root < 0 {
            return 0.0;
        }

        let mut current_idx = self.root as usize;
        let mut depth = 0.0;

        loop {
            let node = self.node_pool.get(current_idx);
            
            if node.is_leaf {
                // Add adjustment for unseen data
                return depth + c_factor(node.size as usize);
            }

            if sample[node.split_feature as usize] < node.split_value {
                if node.left_child < 0 {
                    return depth + c_factor(node.size as usize);
                }
                current_idx = node.left_child as usize;
            } else {
                if node.right_child < 0 {
                    return depth + c_factor(node.size as usize);
                }
                current_idx = node.right_child as usize;
            }

            depth += 1.0;
        }
    }
}

/// C(n) function for path length adjustment
fn c_factor(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    if n == 2 {
        return 1.0;
    }
    
    let ln_n = (n as f64).ln();
    2.0 * (ln_n + 0.5772156649) - 2.0 * (n as f64 - 1.0) / (n as f64)
}

/// Main Isolation Forest with multiple trees
pub struct IsolationForest<const MAX_TREES: usize, const MAX_NODES_PER_TREE: usize> {
    pub trees: [IsolationTree<MAX_NODES_PER_TREE>; MAX_TREES],
    pub n_trees: usize,
    pub threshold: f64,
    pub sample_size: usize,
    pub calculation_counter: AtomicU64,
}

impl<const MAX_TREES: usize, const MAX_NODES_PER_TREE: usize> IsolationForest<MAX_TREES, MAX_NODES_PER_TREE> {
    pub fn new(n_trees: usize, max_depth: u32, threshold: f64) -> Self {
        let trees = std::array::from_fn(|_| IsolationTree::new(max_depth));
        
        IsolationForest {
            trees,
            n_trees: n_trees.min(MAX_TREES),
            threshold,
            sample_size: 256,
            calculation_counter: AtomicU64::new(0),
        }
    }

    /// Fit the forest on training data
    pub fn fit(&mut self, data: &[Vec<f64>]) {
        let sample_size = self.sample_size.min(data.len());
        
        for i in 0..self.n_trees {
            // Subsample data
            let sample: Vec<Vec<f64>> = data.iter()
                .step_by(data.len() / sample_size + 1)
                .take(sample_size)
                .cloned()
                .collect();
            
            self.trees[i].build(&sample, i as u64);
        }
    }

    /// Calculate anomaly score for a sample
    pub fn anomaly_score(&self, sample: &[f64]) -> f64 {
        self.calculation_counter.fetch_add(1, Ordering::Relaxed);
        
        if self.n_trees == 0 {
            return 0.0;
        }

        // Average path length across all trees
        let total_path: f64 = self.trees[..self.n_trees]
            .iter()
            .map(|tree| tree.path_length(sample))
            .sum();
        
        let avg_path = total_path / self.n_trees as f64;
        
        // Anomaly score: s(x, n) = 2^(-E(h(x))/c(n))
        let c_n = c_factor(self.sample_size);
        if c_n == 0.0 {
            return 0.5;
        }
        
        2.0_f64.powf(-avg_path / c_n)
    }

    /// Detect if sample is anomalous
    pub fn is_anomaly(&self, sample: &[f64]) -> bool {
        self.anomaly_score(sample) > self.threshold
    }

    /// Batch anomaly detection
    pub fn detect_batch(&self, samples: &[Vec<f64>]) -> Vec<(usize, f64)> {
        samples.iter()
            .enumerate()
            .filter_map(|(i, sample)| {
                let score = self.anomaly_score(sample);
                if score > self.threshold {
                    Some((i, score))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Update threshold dynamically
    pub fn update_threshold(&mut self, new_threshold: f64) {
        self.threshold = new_threshold.clamp(0.0, 1.0);
    }
}

/// Market microstructure anomaly detector using Isolation Forest
pub struct MarketAnomalyDetector<const MAX_TREES: usize, const MAX_NODES: usize> {
    pub forest: IsolationForest<MAX_TREES, MAX_NODES>,
    pub feature_buffer: Vec<f64>,
    pub anomaly_history: Vec<(u64, f64)>,
}

impl<const MAX_TREES: usize, const MAX_NODES: usize> MarketAnomalyDetector<MAX_TREES, MAX_NODES> {
    pub fn new() -> Self {
        MarketAnomalyDetector {
            forest: IsolationForest::new(10, 8, 0.6),
            feature_buffer: Vec::with_capacity(16),
            anomaly_history: Vec::with_capacity(1024),
        }
    }

    /// Extract features from order book snapshot
    pub fn extract_features(&mut self, bid_prices: &[f64], ask_prices: &[f64], volumes: &[f64]) {
        self.feature_buffer.clear();
        
        // Mid price
        let mid = if !bid_prices.is_empty() && !ask_prices.is_empty() {
            (bid_prices[0] + ask_prices[0]) / 2.0
        } else {
            0.0
        };
        self.feature_buffer.push(mid);
        
        // Spread
        let spread = if !bid_prices.is_empty() && !ask_prices.is_empty() {
            ask_prices[0] - bid_prices[0]
        } else {
            0.0
        };
        self.feature_buffer.push(spread);
        
        // Imbalance
        let bid_vol: f64 = volumes.iter().take(volumes.len() / 2).sum();
        let ask_vol: f64 = volumes.iter().skip(volumes.len() / 2).sum();
        let imbalance = if bid_vol + ask_vol > 0.0 {
            (bid_vol - ask_vol) / (bid_vol + ask_vol)
        } else {
            0.0
        };
        self.feature_buffer.push(imbalance);
        
        // Add more features as needed
    }

    /// Check for market anomaly
    pub fn check_anomaly(&mut self) -> AnomalyResult {
        if self.feature_buffer.is_empty() {
            return AnomalyResult::NoData;
        }

        let score = self.forest.anomaly_score(&self.feature_buffer);
        let is_anomaly = score > self.forest.threshold;

        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        if is_anomaly {
            self.anomaly_history.push((timestamp, score));
        }

        AnomalyResult {
            is_anomaly,
            score,
            anomaly_type: self.classify_anomaly(score),
        }
    }

    fn classify_anomaly(&self, score: f64) -> AnomalyType {
        if score > 0.8 {
            AnomalyType::FlashCrash
        } else if score > 0.7 {
            AnomalyType::Spoofing
        } else if score > 0.6 {
            AnomalyType::LiquidityShock
        } else {
            AnomalyType::Normal
        }
    }
}

impl<const MAX_TREES: usize, const MAX_NODES: usize> Default for MarketAnomalyDetector<MAX_TREES, MAX_NODES> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct AnomalyResult {
    pub is_anomaly: bool,
    pub score: f64,
    pub anomaly_type: AnomalyType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnomalyType {
    Normal,
    FlashCrash,
    Spoofing,
    LiquidityShock,
}

impl AnomalyResult {
    pub fn severity(&self) -> SeverityLevel {
        if !self.is_anomaly {
            SeverityLevel::None
        } else if self.score > 0.8 {
            SeverityLevel::Critical
        } else if self.score > 0.7 {
            SeverityLevel::High
        } else {
            SeverityLevel::Medium
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeverityLevel {
    None,
    Medium,
    High,
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_pool_allocation() {
        let pool: NodePool<100> = NodePool::new();
        
        let idx1 = pool.allocate();
        let idx2 = pool.allocate();
        
        assert!(idx1.is_some());
        assert!(idx2.is_some());
        assert_eq!(idx1.unwrap(), 0);
        assert_eq!(idx2.unwrap(), 1);
        assert_eq!(pool.used_count(), 2);
    }

    #[test]
    fn test_isolation_tree_path_length() {
        let mut tree: IsolationTree<100> = IsolationTree::new(5);
        
        let data = vec![
            vec![1.0, 2.0],
            vec![2.0, 3.0],
            vec![3.0, 4.0],
            vec![4.0, 5.0],
        ];
        
        tree.build(&data, 42);
        
        let path = tree.path_length(&[2.5, 3.5]);
        assert!(path >= 0.0);
    }

    #[test]
    fn test_isolation_forest_anomaly_score() {
        let mut forest: IsolationForest<5, 100> = IsolationForest::new(3, 4, 0.6);
        
        // Training data (normal)
        let train_data: Vec<Vec<f64>> = (0..100)
            .map(|i| vec![i as f64 / 100.0, (i as f64 / 100.0).powi(2)])
            .collect();
        
        forest.fit(&train_data);
        
        // Normal sample should have low score
        let normal_score = forest.anomaly_score(&[0.5, 0.25]);
        
        // Anomalous sample should have higher score
        let anomaly_score = forest.anomaly_score(&[10.0, 100.0]);
        
        assert!(anomaly_score > normal_score);
    }

    #[test]
    fn test_c_factor() {
        assert_eq!(c_factor(0), 0.0);
        assert_eq!(c_factor(1), 0.0);
        assert_eq!(c_factor(2), 1.0);
        assert!(c_factor(100) > 1.0);
    }
}
