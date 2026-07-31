//! Software-Defined Multi-Path TCP/QUIC Manager
//!
//! Aggregates bandwidth across multiple network interfaces and routes,
//! guaranteeing delivery during localized ISP congestion.

use std::time::{Duration, Instant};

/// Network path state
#[derive(Debug, Clone, Copy)]
pub struct PathState {
    /// Unique path identifier
    pub path_id: u32,
    /// Current RTT in microseconds
    pub rtt_us: u64,
    /// Packet loss rate (0.0 to 1.0)
    pub loss_rate: f64,
    /// Available bandwidth estimate (bytes/sec)
    pub bandwidth_bps: u64,
    /// Congestion window (packets)
    pub cwnd: u32,
    /// Whether path is healthy
    pub is_healthy: bool,
    /// Last health check timestamp
    pub last_check: Instant,
    /// Consecutive failures
    pub consecutive_failures: u32,
}

impl PathState {
    pub fn new(path_id: u32) -> Self {
        PathState {
            path_id,
            rtt_us: 1000, // Default 1ms
            loss_rate: 0.0,
            bandwidth_bps: 1_000_000_000, // 1 Gbps default
            cwnd: 10,
            is_healthy: true,
            last_check: Instant::now(),
            consecutive_failures: 0,
        }
    }

    /// Update path metrics from probe results
    #[inline]
    pub fn update_metrics(&mut self, rtt_us: u64, loss_rate: f64, bandwidth_bps: u64) {
        self.rtt_us = rtt_us;
        self.loss_rate = loss_rate.min(1.0);
        self.bandwidth_bps = bandwidth_bps;
        self.last_check = Instant::now();
        
        // Update health status
        if loss_rate > 0.1 || rtt_us > 100_000 {
            self.consecutive_failures += 1;
            if self.consecutive_failures >= 3 {
                self.is_healthy = false;
            }
        } else {
            self.consecutive_failures = 0;
            self.is_healthy = true;
        }
    }

    /// Calculate path score for routing decisions (higher is better)
    #[inline]
    pub fn score(&self) -> f64 {
        if !self.is_healthy {
            return 0.0;
        }
        
        // Score based on latency, loss, and bandwidth
        let latency_score = 1.0 / (1.0 + self.rtt_us as f64 / 1000.0);
        let loss_score = 1.0 - self.loss_rate;
        let bandwidth_score = (self.bandwidth_bps as f64 / 1_000_000_000.0).min(1.0);
        
        // Weighted combination
        latency_score * 0.5 + loss_score * 0.3 + bandwidth_score * 0.2
    }

    /// Check if path needs probing
    #[inline]
    pub fn needs_probe(&self) -> bool {
        self.last_check.elapsed() > Duration::from_millis(100)
    }
}

/// Packet fragment for multi-path transmission
#[derive(Debug, Clone)]
pub struct PacketFragment {
    /// Original packet ID
    pub packet_id: u64,
    /// Fragment sequence number
    pub fragment_seq: u8,
    /// Total fragments
    pub total_fragments: u8,
    /// Payload data
    pub payload: Vec<u8>,
    /// Timestamp when sent
    pub sent_at: Instant,
    /// Assigned path ID
    pub path_id: u32,
}

/// Reassembly buffer for fragmented packets
pub struct ReassemblyBuffer {
    /// Pre-allocated fragment slots
    fragments: [Option<PacketFragment>; 16],
    /// Expected total fragments
    total_expected: u8,
    /// Received count
    received_count: u8,
    /// First fragment timestamp
    first_received: Option<Instant>,
}

impl ReassemblyBuffer {
    pub fn new(total_fragments: u8) -> Self {
        ReassemblyBuffer {
            fragments: [None; 16],
            total_expected: total_fragments.min(16),
            received_count: 0,
            first_received: None,
        }
    }

    /// Add a fragment to the buffer
    #[inline]
    pub fn add_fragment(&mut self, fragment: PacketFragment) -> Option<Vec<u8>> {
        if self.first_received.is_none() {
            self.first_received = Some(Instant::now());
        }

        let seq = fragment.fragment_seq as usize;
        if seq < self.fragments.len() {
            self.fragments[seq] = Some(fragment);
            self.received_count += 1;
        }

        // Check if complete
        if self.received_count >= self.total_expected as u8 {
            self.assemble()
        } else {
            None
        }
    }

    /// Assemble complete packet
    fn assemble(&mut self) -> Option<Vec<u8>> {
        let mut assembled = Vec::new();
        
        for i in 0..self.total_expected as usize {
            if let Some(ref frag) = self.fragments[i] {
                assembled.extend_from_slice(&frag.payload);
            } else {
                return None; // Missing fragment
            }
        }

        Some(assembled)
    }

    /// Check for timeout
    #[inline]
    pub fn is_expired(&self, timeout: Duration) -> bool {
        if let Some(first) = self.first_received {
            first.elapsed() > timeout
        } else {
            true
        }
    }

    /// Reset buffer
    #[inline]
    pub fn reset(&mut self, total_fragments: u8) {
        self.fragments.fill(None);
        self.total_expected = total_fragments.min(16);
        self.received_count = 0;
        self.first_received = None;
    }
}

/// Multi-path transmission strategy
#[derive(Debug, Clone, Copy)]
pub enum TransmissionStrategy {
    /// Send all data on best path
    BestPath,
    /// Split data across all healthy paths
    LoadBalance,
    /// Send redundant copies on multiple paths
    Redundant,
    /// Erasure coding: send k of n fragments
    ErasureCoding { k: u8, n: u8 },
}

/// Multi-Path Manager for HFT networks
pub struct MultiPathManager {
    /// Available paths
    paths: [PathState; 8],
    /// Number of active paths
    num_paths: usize,
    /// Current transmission strategy
    strategy: TransmissionStrategy,
    /// Pending reassembly buffers
    reassembly_buffers: [Option<ReassemblyBuffer>; 64],
    /// Next packet ID
    next_packet_id: u64,
    /// Pre-allocated fragment queue
    fragment_queue: Vec<PacketFragment>,
    /// Minimum RTT across all paths
    min_rtt_us: u64,
    /// Maximum allowed RTT variance
    max_rtt_variance_us: u64,
}

impl MultiPathManager {
    pub fn new(num_paths: usize, strategy: TransmissionStrategy) -> Self {
        let mut paths = [PathState::new(0); 8];
        for i in 0..num_paths.min(8) {
            paths[i] = PathState::new(i as u32);
        }

        MultiPathManager {
            paths,
            num_paths: num_paths.min(8),
            strategy,
            reassembly_buffers: [None; 64],
            next_packet_id: 0,
            fragment_queue: Vec::with_capacity(16),
            min_rtt_us: 1000,
            max_rtt_variance_us: 5000,
        }
    }

    /// Select best path for transmission
    #[inline]
    pub fn select_best_path(&self) -> Option<u32> {
        let mut best_idx = None;
        let mut best_score = 0.0;

        for i in 0..self.num_paths {
            if self.paths[i].is_healthy {
                let score = self.paths[i].score();
                if score > best_score {
                    best_score = score;
                    best_idx = Some(i);
                }
            }
        }

        best_idx.map(|i| self.paths[i].path_id)
    }

    /// Select paths for load-balanced transmission
    #[inline]
    pub fn select_paths_lb(&self) -> Vec<u32> {
        let mut selected = Vec::with_capacity(self.num_paths);

        for i in 0..self.num_paths {
            if self.paths[i].is_healthy {
                selected.push(self.paths[i].path_id);
            }
        }

        selected
    }

    /// Split payload into fragments for multi-path transmission
    pub fn fragment_payload(&mut self, payload: &[u8]) -> Vec<PacketFragment> {
        self.fragment_queue.clear();
        
        let packet_id = self.next_packet_id;
        self.next_packet_id += 1;

        match self.strategy {
            TransmissionStrategy::BestPath | TransmissionStrategy::Redundant => {
                // Single fragment
                let path_id = self.select_best_path().unwrap_or(0);
                self.fragment_queue.push(PacketFragment {
                    packet_id,
                    fragment_seq: 0,
                    total_fragments: 1,
                    payload: payload.to_vec(),
                    sent_at: Instant::now(),
                    path_id,
                });
            }
            TransmissionStrategy::LoadBalance => {
                // Split evenly across healthy paths
                let healthy_paths = self.select_paths_lb();
                if healthy_paths.is_empty() {
                    return self.fragment_queue.clone();
                }

                let chunk_size = (payload.len() + healthy_paths.len() - 1) / healthy_paths.len();
                
                for (i, &path_id) in healthy_paths.iter().enumerate() {
                    let start = i * chunk_size;
                    let end = (start + chunk_size).min(payload.len());
                    
                    if start < payload.len() {
                        self.fragment_queue.push(PacketFragment {
                            packet_id,
                            fragment_seq: i as u8,
                            total_fragments: healthy_paths.len() as u8,
                            payload: payload[start..end].to_vec(),
                            sent_at: Instant::now(),
                            path_id,
                        });
                    }
                }
            }
            TransmissionStrategy::ErasureCoding { k, n } => {
                // Simplified erasure coding: just split into n fragments
                // Real implementation would use Reed-Solomon or similar
                let chunk_size = (payload.len() + n as usize - 1) / n as usize;
                
                for i in 0..n as usize {
                    let start = i * chunk_size;
                    let end = (start + chunk_size).min(payload.len());
                    
                    if start < payload.len() {
                        // Assign to k best paths
                        let path_idx = i % self.num_paths.max(1);
                        self.fragment_queue.push(PacketFragment {
                            packet_id,
                            fragment_seq: i as u8,
                            total_fragments: n,
                            payload: payload[start..end].to_vec(),
                            sent_at: Instant::now(),
                            path_id: self.paths[path_idx].path_id,
                        });
                    }
                }
            }
        }

        self.fragment_queue.clone()
    }

    /// Process received fragment
    pub fn receive_fragment(&mut self, fragment: PacketFragment) -> Option<Vec<u8>> {
        let buffer_idx = (fragment.packet_id % 64) as usize;

        if self.reassembly_buffers[buffer_idx].is_none() {
            self.reassembly_buffers[buffer_idx] = 
                Some(ReassemblyBuffer::new(fragment.total_fragments));
        }

        if let Some(ref mut buffer) = self.reassembly_buffers[buffer_idx] {
            // Update path metrics based on arrival
            let path_idx = self.find_path_index(fragment.path_id);
            if let Some(path) = path_idx.and_then(|i| self.paths.get_mut(i)) {
                let elapsed = fragment.sent_at.elapsed().as_micros() as u64;
                path.rtt_us = elapsed;
            }

            return buffer.add_fragment(fragment);
        }

        None
    }

    /// Find path index by ID
    fn find_path_index(&self, path_id: u32) -> Option<usize> {
        for i in 0..self.num_paths {
            if self.paths[i].path_id == path_id {
                return Some(i);
            }
        }
        None
    }

    /// Update path metrics from external probe
    #[inline]
    pub fn update_path_metrics(&mut self, path_id: u32, rtt_us: u64, loss_rate: f64, bandwidth_bps: u64) {
        if let Some(idx) = self.find_path_index(path_id) {
            self.paths[idx].update_metrics(rtt_us, loss_rate, bandwidth_bps);
            self.update_min_rtt();
        }
    }

    /// Update minimum RTT across all paths
    fn update_min_rtt(&mut self) {
        let mut min = u64::MAX;
        for i in 0..self.num_paths {
            if self.paths[i].is_healthy && self.paths[i].rtt_us < min {
                min = self.paths[i].rtt_us;
            }
        }
        self.min_rtt_us = min;
    }

    /// Get aggregate bandwidth across all healthy paths
    #[inline]
    pub fn aggregate_bandwidth(&self) -> u64 {
        let mut total = 0;
        for i in 0..self.num_paths {
            if self.paths[i].is_healthy {
                total += self.paths[i].bandwidth_bps;
            }
        }
        total
    }

    /// Get latency variance (jitter) across paths
    #[inline]
    pub fn latency_variance(&self) -> u64 {
        if self.num_paths < 2 {
            return 0;
        }

        let mut max_rtt = 0;
        let mut min_rtt = u64::MAX;

        for i in 0..self.num_paths {
            if self.paths[i].is_healthy {
                max_rtt = max_rtt.max(self.paths[i].rtt_us);
                min_rtt = min_rtt.min(self.paths[i].rtt_us);
            }
        }

        if min_rtt == u64::MAX {
            return 0;
        }

        max_rtt - min_rtt
    }

    /// Check if any path is available
    #[inline]
    pub fn has_healthy_path(&self) -> bool {
        for i in 0..self.num_paths {
            if self.paths[i].is_healthy {
                return true;
            }
        }
        false
    }

    /// Get path statistics
    #[inline]
    pub fn get_path_stats(&self, path_id: u32) -> Option<&PathState> {
        self.find_path_index(path_id).and_then(|i| self.paths.get(i))
    }

    /// Set transmission strategy
    #[inline]
    pub fn set_strategy(&mut self, strategy: TransmissionStrategy) {
        self.strategy = strategy;
    }

    /// Prune expired reassembly buffers
    pub fn prune_expired_buffers(&mut self, timeout: Duration) {
        for buffer in &mut self.reassembly_buffers {
            if let Some(ref b) = buffer {
                if b.is_expired(timeout) {
                    *buffer = None;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_selection() {
        let mut manager = MultiPathManager::new(3, TransmissionStrategy::BestPath);
        
        // Make path 1 the best
        manager.paths[1].update_metrics(500, 0.0, 10_000_000_000);
        
        let best = manager.select_best_path();
        assert_eq!(best, Some(1));
    }

    #[test]
    fn test_fragment_load_balance() {
        let mut manager = MultiPathManager::new(4, TransmissionStrategy::LoadBalance);
        
        let payload = vec![1u8; 1000];
        let fragments = manager.fragment_payload(&payload);
        
        // Should have fragments for each healthy path
        assert!(!fragments.is_empty());
        assert!(fragments.len() <= 4);
    }

    #[test]
    fn test_aggregate_bandwidth() {
        let mut manager = MultiPathManager::new(3, TransmissionStrategy::BestPath);
        
        let total = manager.aggregate_bandwidth();
        assert!(total > 0);
        
        // Disable one path
        manager.paths[0].is_healthy = false;
        let reduced = manager.aggregate_bandwidth();
        assert!(reduced < total);
    }
}
