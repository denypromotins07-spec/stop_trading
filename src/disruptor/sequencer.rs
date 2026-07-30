//! Sequencer and Sequence Barrier Implementation
//!
//! Builds a sequence barrier and dependency graph tracker for event processors.
//! Ensures consumers wait efficiently using adaptive spinning (yielding to OS
//! only after threshold) to minimize wake-up latency.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

/// Wait strategy for consumers
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitStrategy {
    /// Busy spin - highest CPU usage, lowest latency
    BusySpin,
    /// Spin then yield after threshold
    AdaptiveSpin,
    /// Yield on every wait
    Yielding,
    /// Block with parking
    Blocking,
    /// Timeout-based waiting
    Timeout { timeout_ns: u64 },
}

impl Default for WaitStrategy {
    fn default() -> Self {
        WaitStrategy::AdaptiveSpin
    }
}

/// Sequence barrier for coordinating producers and consumers
#[repr(C)]
pub struct SequenceBarrier {
    /// Cursor to wait for
    cursor: *const AtomicU64,
    /// Dependent sequences (other barriers this depends on)
    dependencies: Vec<u64>,
    /// Alert flag for shutdown
    alerted: AtomicBool,
    /// Wait strategy
    wait_strategy: WaitStrategy,
    /// Spin count before yielding
    spin_threshold: u32,
    /// Current spin count
    current_spins: AtomicU32,
    /// Padding for cache alignment
    _padding: [u8; 64 - std::mem::size_of::<*const AtomicU64>() - 8 - 1 - 1 - 4 - 4],
}

// Safety: SequenceBarrier is safe to share when properly constructed
unsafe impl Send for SequenceBarrier {}
unsafe impl Sync for SequenceBarrier {}

use std::sync::atomic::AtomicU32;

impl SequenceBarrier {
    /// Create a new sequence barrier
    pub fn new(cursor: &AtomicU64, wait_strategy: WaitStrategy) -> Self {
        Self {
            cursor: cursor as *const AtomicU64,
            dependencies: Vec::new(),
            alerted: AtomicBool::new(false),
            wait_strategy,
            spin_threshold: 100, // Spin 100 times before yielding
            current_spins: AtomicU32::new(0),
            _padding: [0u8; 64 - std::mem::size_of::<*const AtomicU64>() - 8 - 1 - 1 - 4 - 4],
        }
    }

    /// Add dependency on another sequence
    #[inline]
    pub fn add_dependency(&mut self, seq: u64) {
        self.dependencies.push(seq);
    }

    /// Clear all dependencies
    #[inline]
    pub fn clear_dependencies(&mut self) {
        self.dependencies.clear();
    }

    /// Wait for sequence to be available
    #[inline]
    pub fn wait_for(&self, mut sequence: u64) -> Result<u64, BarrierError> {
        if self.alerted.load(Ordering::Acquire) {
            return Err(BarrierError::Alerted);
        }

        let mut spins = 0u32;

        loop {
            // Check alert status
            if self.alerted.load(Ordering::Acquire) {
                return Err(BarrierError::Alerted);
            }

            // Get current cursor value
            let cursor = unsafe { (*self.cursor).load(Ordering::Acquire) };

            // Find minimum available sequence considering dependencies
            let mut available = cursor;
            for &dep in &self.dependencies {
                if dep < available {
                    available = dep;
                }
            }

            if available >= sequence {
                self.current_spins.store(0, Ordering::Relaxed);
                return Ok(available);
            }

            // Apply wait strategy
            match self.wait_strategy {
                WaitStrategy::BusySpin => {
                    std::hint::spin_loop();
                }
                WaitStrategy::AdaptiveSpin => {
                    spins += 1;
                    if spins < self.spin_threshold {
                        std::hint::spin_loop();
                    } else {
                        thread::yield_now();
                    }
                }
                WaitStrategy::Yielding => {
                    thread::yield_now();
                }
                WaitStrategy::Blocking => {
                    thread::park_timeout(std::time::Duration::from_micros(1));
                }
                WaitStrategy::Timeout { timeout_ns } => {
                    // In production, would track elapsed time
                    if spins > 1000 {
                        return Err(BarrierError::Timeout);
                    }
                    std::hint::spin_loop();
                }
            }

            spins += 1;
            self.current_spins.store(spins, Ordering::Relaxed);
        }
    }

    /// Try to wait without blocking
    #[inline]
    pub fn try_wait_for(&self, sequence: u64) -> Result<u64, BarrierError> {
        if self.alerted.load(Ordering::Acquire) {
            return Err(BarrierError::Alerted);
        }

        let cursor = unsafe { (*self.cursor).load(Ordering::Acquire) };

        let mut available = cursor;
        for &dep in &self.dependencies {
            if dep < available {
                available = dep;
            }
        }

        if available >= sequence {
            Ok(available)
        } else {
            Err(BarrierError::NotAvailable)
        }
    }

    /// Alert the barrier (signal shutdown)
    #[inline]
    pub fn alert(&self) {
        self.alerted.store(true, Ordering::Release);
    }

    /// Reset the alert
    #[inline]
    pub fn reset_alert(&self) {
        self.alerted.store(false, Ordering::Release);
    }

    /// Check if alerted
    #[inline]
    pub fn is_alerted(&self) -> bool {
        self.alerted.load(Ordering::Acquire)
    }

    /// Get current spin count
    #[inline]
    pub fn get_spin_count(&self) -> u32 {
        self.current_spins.load(Ordering::Relaxed)
    }

    /// Set spin threshold
    #[inline]
    pub fn set_spin_threshold(&mut self, threshold: u32) {
        self.spin_threshold = threshold;
    }
}

/// Barrier error types
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarrierError {
    /// Barrier was alerted (shutdown)
    Alerted,
    /// Sequence not yet available
    NotAvailable,
    /// Timeout occurred
    Timeout,
    /// Invalid state
    InvalidState,
}

/// Dependency graph for tracking consumer relationships
#[repr(C)]
pub struct DependencyGraph {
    /// Dependencies stored as adjacency list
    dependencies: Vec<Vec<usize>>,
    /// Number of nodes
    node_count: usize,
    /// Is running
    is_running: AtomicBool,
}

impl DependencyGraph {
    /// Create a new dependency graph
    pub fn new(node_count: usize) -> Self {
        Self {
            dependencies: vec![Vec::new(); node_count],
            node_count,
            is_running: AtomicBool::new(true),
        }
    }

    /// Add dependency: node depends on depends_on
    #[inline]
    pub fn add_dependency(&mut self, node: usize, depends_on: usize) {
        if node < self.node_count && depends_on < self.node_count {
            self.dependencies[node].push(depends_on);
        }
    }

    /// Remove all dependencies for a node
    #[inline]
    pub fn clear_dependencies(&mut self, node: usize) {
        if node < self.node_count {
            self.dependencies[node].clear();
        }
    }

    /// Get all nodes that a node depends on
    #[inline]
    pub fn get_dependencies(&self, node: usize) -> &[usize] {
        if node < self.node_count {
            &self.dependencies[node]
        } else {
            &[]
        }
    }

    /// Check if adding an edge would create a cycle
    #[inline]
    pub fn would_create_cycle(&self, from: usize, to: usize) -> bool {
        if from == to {
            return true;
        }

        // DFS to check if we can reach 'from' starting from 'to'
        let mut visited = vec![false; self.node_count];
        self.has_path(to, from, &mut visited)
    }

    /// Depth-first search for path existence
    fn has_path(&self, from: usize, to: usize, visited: &mut [bool]) -> bool {
        if from == to {
            return true;
        }

        if visited[from] {
            return false;
        }

        visited[from] = true;

        for &dep in &self.dependencies[from] {
            if self.has_path(dep, to, visited) {
                return true;
            }
        }

        false
    }

    /// Get topological order of nodes
    #[inline]
    pub fn topological_sort(&self) -> Option<Vec<usize>> {
        let mut result = Vec::with_capacity(self.node_count);
        let mut visited = vec![false; self.node_count];
        let mut temp_mark = vec![false; self.node_count];

        for i in 0..self.node_count {
            if !visited[i] {
                if !self.visit(i, &mut visited, &mut temp_mark, &mut result) {
                    return None; // Cycle detected
                }
            }
        }

        result.reverse();
        Some(result)
    }

    fn visit(&self, node: usize, visited: &mut [bool], temp_mark: &mut [bool], result: &mut Vec<usize>) -> bool {
        if temp_mark[node] {
            return false; // Cycle detected
        }

        if visited[node] {
            return true;
        }

        temp_mark[node] = true;

        for &dep in &self.dependencies[node] {
            if !self.visit(dep, visited, temp_mark, result) {
                return false;
            }
        }

        temp_mark[node] = false;
        visited[node] = true;
        result.push(node);
        true
    }

    /// Stop the graph
    #[inline]
    pub fn stop(&self) {
        self.is_running.store(false, Ordering::Release);
    }

    /// Check if running
    #[inline]
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Acquire)
    }
}

/// Sequencer for coordinating event publication
#[repr(C)]
pub struct Sequencer {
    /// Current cursor position
    cursor: AtomicU64,
    /// Next claimable sequence
    next_sequence: AtomicU64,
    /// Buffer size mask
    buffer_mask: usize,
    /// Wait strategy
    wait_strategy: WaitStrategy,
    /// Is running
    is_running: AtomicBool,
    /// Pending publication count
    pending_count: AtomicU64,
    /// Total published events
    total_published: AtomicU64,
    /// Backpressure events
    backpressure_events: AtomicU64,
}

impl Sequencer {
    /// Create a new sequencer
    pub fn new(wait_strategy: WaitStrategy) -> Self {
        Self {
            cursor: AtomicU64::new(0),
            next_sequence: AtomicU64::new(1),
            buffer_mask: 1023, // Default for 1024-size buffer
            wait_strategy,
            is_running: AtomicBool::new(false),
            pending_count: AtomicU64::new(0),
            total_published: AtomicU64::new(0),
            backpressure_events: AtomicU64::new(0),
        }
    }

    /// Set buffer mask
    #[inline]
    pub fn set_buffer_mask(&mut self, mask: usize) {
        self.buffer_mask = mask;
    }

    /// Start the sequencer
    #[inline]
    pub fn start(&self) {
        self.is_running.store(true, Ordering::Release);
    }

    /// Stop the sequencer
    #[inline]
    pub fn stop(&self) {
        self.is_running.store(false, Ordering::Release);
    }

    /// Check if running
    #[inline]
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Acquire)
    }

    /// Claim next sequence number (blocking)
    #[inline]
    pub fn next(&self, batch_size: u64) -> Result<u64, ()> {
        if !self.is_running.load(Ordering::Acquire) {
            return Err(());
        }

        let sequence = self.next_sequence.fetch_add(batch_size, Ordering::AcqRel);
        self.pending_count.fetch_add(batch_size, Ordering::Relaxed);
        Ok(sequence)
    }

    /// Try to claim next sequence (non-blocking)
    #[inline]
    pub fn try_next(&self, batch_size: u64) -> Result<u64, ()> {
        if !self.is_running.load(Ordering::Acquire) {
            return Err(());
        }

        // Check if we have capacity (simplified check)
        let pending = self.pending_count.load(Ordering::Relaxed);
        if pending > self.buffer_mask as u64 {
            self.backpressure_events.fetch_add(1, Ordering::Relaxed);
            return Err(());
        }

        let sequence = self.next_sequence.fetch_add(batch_size, Ordering::AcqRel);
        self.pending_count.fetch_add(batch_size, Ordering::Relaxed);
        Ok(sequence)
    }

    /// Publish a sequence (make it available to consumers)
    #[inline]
    pub fn publish(&self, sequence: u64) {
        // Update cursor to indicate this sequence is ready
        self.cursor.store(sequence, Ordering::Release);
        
        // Decrement pending count
        let pending = self.pending_count.fetch_sub(1, Ordering::Relaxed);
        if pending > 0 {
            self.pending_count.store(pending - 1, Ordering::Relaxed);
        }

        self.total_published.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current cursor position
    #[inline]
    pub fn cursor(&self) -> u64 {
        self.cursor.load(Ordering::Acquire)
    }

    /// Get available sequence for a dependent consumer
    #[inline]
    pub fn get_available_sequence(&self, dependent_seq: u64) -> u64 {
        let cursor = self.cursor.load(Ordering::Acquire);
        
        if cursor > dependent_seq {
            cursor
        } else {
            dependent_seq
        }
    }

    /// Create a new sequence barrier
    #[inline]
    pub fn create_barrier(&self) -> SequenceBarrier {
        SequenceBarrier::new(&self.cursor, self.wait_strategy)
    }

    /// Get pending count
    #[inline]
    pub fn pending_count(&self) -> u64 {
        self.pending_count.load(Ordering::Relaxed)
    }

    /// Get total published count
    #[inline]
    pub fn total_published(&self) -> u64 {
        self.total_published.load(Ordering::Relaxed)
    }

    /// Get backpressure event count
    #[inline]
    pub fn backpressure_events(&self) -> u64 {
        self.backpressure_events.load(Ordering::Relaxed)
    }

    /// Get sequencer statistics
    #[inline]
    pub fn get_stats(&self) -> SequencerStats {
        SequencerStats {
            cursor: self.cursor(),
            next_sequence: self.next_sequence.load(Ordering::Relaxed),
            pending_count: self.pending_count(),
            total_published: self.total_published(),
            backpressure_events: self.backpressure_events(),
            is_running: self.is_running(),
        }
    }
}

/// Sequencer statistics
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SequencerStats {
    pub cursor: u64,
    pub next_sequence: u64,
    pub pending_count: u64,
    pub total_published: u64,
    pub backpressure_events: u64,
    pub is_running: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequencer_creation() {
        let seq = Sequencer::new(WaitStrategy::default());
        
        assert!(!seq.is_running());
        assert_eq!(seq.cursor(), 0);
        assert_eq!(seq.pending_count(), 0);
    }

    #[test]
    fn test_sequencer_lifecycle() {
        let seq = Sequencer::new(WaitStrategy::default());
        
        seq.start();
        assert!(seq.is_running());
        
        seq.stop();
        assert!(!seq.is_running());
    }

    #[test]
    fn test_sequence_claim() {
        let seq = Sequencer::new(WaitStrategy::default());
        seq.start();
        
        let s1 = seq.next(1).unwrap();
        let s2 = seq.next(1).unwrap();
        
        assert_eq!(s1, 1);
        assert_eq!(s2, 2);
        
        seq.publish(s1);
        assert_eq!(seq.cursor(), 1);
    }

    #[test]
    fn test_sequence_barrier() {
        use std::sync::atomic::AtomicU64;
        
        let cursor = AtomicU64::new(5);
        let barrier = SequenceBarrier::new(&cursor, WaitStrategy::BusySpin);
        
        // Should immediately return since cursor is already at 5
        let result = barrier.try_wait_for(3);
        assert!(result.is_ok());
        assert!(result.unwrap() >= 3);
        
        // Should fail since cursor is at 5, waiting for 10
        let result = barrier.try_wait_for(10);
        assert!(result.is_err());
    }

    #[test]
    fn test_dependency_graph() {
        let mut graph = DependencyGraph::new(4);
        
        // A -> B -> C -> D
        graph.add_dependency(0, 1); // 0 depends on 1
        graph.add_dependency(1, 2); // 1 depends on 2
        graph.add_dependency(2, 3); // 2 depends on 3
        
        assert_eq!(graph.get_dependencies(0), &[1]);
        assert_eq!(graph.get_dependencies(1), &[2]);
        
        // Test cycle detection
        assert!(!graph.would_create_cycle(0, 3)); // Adding 0->3 is fine
        assert!(graph.would_create_cycle(3, 0)); // Adding 3->0 would create cycle
    }

    #[test]
    fn test_topological_sort() {
        let mut graph = DependencyGraph::new(3);
        
        // Linear dependency: 0 -> 1 -> 2
        graph.add_dependency(0, 1);
        graph.add_dependency(1, 2);
        
        let order = graph.topological_sort();
        assert!(order.is_some());
        
        let order = order.unwrap();
        // 2 should come before 1, 1 before 0
        assert!(order.iter().position(|&x| x == 2).unwrap() < order.iter().position(|&x| x == 1).unwrap());
        assert!(order.iter().position(|&x| x == 1).unwrap() < order.iter().position(|&x| x == 0).unwrap());
    }

    #[test]
    fn test_barrier_alert() {
        let cursor = AtomicU64::new(0);
        let barrier = SequenceBarrier::new(&cursor, WaitStrategy::BusySpin);
        
        assert!(!barrier.is_alerted());
        
        barrier.alert();
        assert!(barrier.is_alerted());
        
        barrier.reset_alert();
        assert!(!barrier.is_alerted());
    }

    #[test]
    fn test_sequencer_stats() {
        let seq = Sequencer::new(WaitStrategy::default());
        seq.start();
        
        let _ = seq.next(1);
        seq.publish(1);
        
        let stats = seq.get_stats();
        assert_eq!(stats.total_published, 1);
        assert!(stats.is_running);
    }
}
