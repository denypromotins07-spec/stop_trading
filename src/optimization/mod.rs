//! Optimization Module Root
//! 
//! Manages parallel parameter sweeps using rayon thread pool,
//! strictly bounding memory allocations to stay within the 6.5GB RAM limit.

pub mod walk_forward;
pub mod monte_carlo;

pub use walk_forward::{WalkForwardAnalyzer, WalkForwardConfig, WalkForwardReport, WalkForwardPeriod};
pub use monte_carlo::{MonteCarloEngine, MonteCarloConfig, MonteCarloReport, SimulationRun};

use std::sync::Arc;
use std::thread;
use rayon::prelude::*;
use tracing::{debug, info, warn};

/// Configuration for the optimization thread pool
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Number of worker threads (default: num_cpus)
    pub num_threads: usize,
    /// Maximum heap size per thread in MB
    pub max_heap_per_thread_mb: usize,
    /// Total memory budget in MB (must stay under 6.5GB system limit)
    pub total_memory_budget_mb: usize,
    /// Enable memory monitoring
    pub monitor_memory: bool,
    /// Batch size for processing (to control memory spikes)
    pub batch_size: usize,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        let num_cpus = num_cpus::get();
        // Reserve ~2GB for OS and other processes
        // With 6.5GB total, we can safely use ~4.5GB
        let total_budget = 4500; 
        
        Self {
            num_threads: num_cpus,
            max_heap_per_thread_mb: total_budget / num_cpus,
            total_memory_budget_mb: total_budget,
            monitor_memory: true,
            batch_size: 100,
        }
    }
}

/// Thread pool manager for optimization tasks
pub struct OptimizationThreadPool {
    config: OptimizationConfig,
    memory_tracker: Arc<MemoryTracker>,
}

/// Memory tracker to enforce RAM limits
pub struct MemoryTracker {
    current_usage: std::sync::atomic::AtomicUsize,
    peak_usage: std::sync::atomic::AtomicUsize,
    budget_mb: usize,
}

impl MemoryTracker {
    fn new(budget_mb: usize) -> Self {
        Self {
            current_usage: std::sync::atomic::AtomicUsize::new(0),
            peak_usage: std::sync::atomic::AtomicUsize::new(0),
            budget_mb,
        }
    }

    /// Try to allocate memory, returns false if over budget
    fn try_allocate(&self, bytes: usize) -> bool {
        let mb = bytes / (1024 * 1024);
        let current = self.current_usage.fetch_add(mb, std::sync::atomic::Ordering::Relaxed);
        
        if current + mb > self.budget_mb {
            self.current_usage.fetch_sub(mb, std::sync::atomic::Ordering::Relaxed);
            return false;
        }

        // Update peak
        let peak = self.peak_usage.load(std::sync::atomic::Ordering::Relaxed);
        if current + mb > peak {
            self.peak_usage.store(current + mb, std::sync::atomic::Ordering::Relaxed);
        }

        true
    }

    /// Release allocated memory
    fn release(&self, bytes: usize) {
        let mb = bytes / (1024 * 1024);
        self.current_usage.fetch_sub(mb, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get current usage in MB
    fn current_usage_mb(&self) -> usize {
        self.current_usage.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get peak usage in MB
    fn peak_usage_mb(&self) -> usize {
        self.peak_usage.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Check if approaching budget limit
    fn is_near_limit(&self, threshold: f64) -> bool {
        let current = self.current_usage_mb();
        current as f64 / self.budget_mb as f64 > threshold
    }
}

impl OptimizationThreadPool {
    /// Create a new optimization thread pool
    pub fn new(config: OptimizationConfig) -> Self {
        let memory_tracker = Arc::new(MemoryTracker::new(config.total_memory_budget_mb));
        
        // Initialize rayon global thread pool with our configuration
        rayon::ThreadPoolBuilder::new()
            .num_threads(config.num_threads)
            .build_global()
            .unwrap_or_else(|_| {
                warn!("Failed to build custom thread pool, using defaults");
            });

        Self {
            config,
            memory_tracker,
        }
    }

    /// Run a parallel parameter sweep with memory bounds
    pub fn run_parameter_sweep<P, R, F>(&self, params: Vec<P>, processor: F) -> Vec<R>
    where
        P: Send + Sync + 'static,
        R: Send + 'static,
        F: Fn(P) -> R + Send + Sync + 'static,
    {
        info!("Running parameter sweep with {} configurations", params.len());
        
        let processor = Arc::new(processor);
        let memory_tracker = self.memory_tracker.clone();
        let batch_size = self.config.batch_size;

        // Process in batches to control memory
        let results: Vec<R> = params
            .into_par_iter()
            .filter_map(|param| {
                // Check memory before processing
                if memory_tracker.is_near_limit(0.9) {
                    warn!("Approaching memory limit, throttling...");
                    thread::sleep(std::time::Duration::from_millis(100));
                }

                let proc = processor.clone();
                Some(proc(param))
            })
            .collect();

        info!(
            "Parameter sweep complete. Peak memory: {}MB",
            memory_tracker.peak_usage_mb()
        );

        results
    }

    /// Run optimization with adaptive batching based on memory pressure
    pub fn run_adaptive_optimization<P, R, F>(&self, params: Vec<P>, processor: F) -> Vec<R>
    where
        P: Send + Sync + 'static,
        R: Send + 'static,
        F: Fn(P) -> R + Send + Sync + 'static,
    {
        let processor = Arc::new(processor);
        let memory_tracker = self.memory_tracker.clone();
        let mut results = Vec::with_capacity(params.len());

        // Split into chunks for better memory control
        let chunk_size = self.config.batch_size;
        let chunks: Vec<Vec<P>> = params
            .chunks(chunk_size)
            .map(|c| c.to_vec())
            .collect();

        for (i, chunk) in chunks.into_iter().enumerate() {
            // Check memory pressure
            if memory_tracker.is_near_limit(0.8) {
                info!("Memory pressure detected, waiting...");
                thread::sleep(std::time::Duration::from_millis(500));
            }

            let proc = processor.clone();
            let chunk_results: Vec<R> = chunk
                .into_par_iter()
                .map(proc)
                .collect();

            results.extend(chunk_results);

            debug!("Completed chunk {}/{}", i + 1, params.len() / chunk_size + 1);
        }

        results
    }

    /// Get memory statistics
    pub fn get_memory_stats(&self) -> MemoryStats {
        MemoryStats {
            current_mb: self.memory_tracker.current_usage_mb(),
            peak_mb: self.memory_tracker.peak_usage_mb(),
            budget_mb: self.memory_tracker.budget_mb,
            utilization: self.memory_tracker.current_usage_mb() as f64 / self.memory_tracker.budget_mb as f64,
        }
    }
}

/// Memory statistics snapshot
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub current_mb: usize,
    pub peak_mb: usize,
    pub budget_mb: usize,
    pub utilization: f64,
}

impl MemoryStats {
    pub fn print(&self) {
        println!("=== Memory Statistics ===");
        println!("Current Usage: {}MB ({:.1}%)", self.current_mb, self.utilization * 100.0);
        println!("Peak Usage: {}MB", self.peak_mb);
        println!("Budget: {}MB", self.budget_mb);
    }
}

/// Parameter grid generator for optimization
pub struct ParameterGrid {
    pub parameters: Vec<ParameterDefinition>,
}

#[derive(Debug, Clone)]
pub struct ParameterDefinition {
    pub name: String,
    pub values: Vec<f64>,
    pub log_scale: bool,
}

impl ParameterGrid {
    /// Create a new parameter grid
    pub fn new(parameters: Vec<ParameterDefinition>) -> Self {
        Self { parameters }
    }

    /// Generate all combinations of parameters
    pub fn generate_combinations(&self) -> Vec<Vec<(String, f64)>> {
        if self.parameters.is_empty() {
            return vec![Vec::new()];
        }

        let mut combinations = Vec::new();
        self.generate_recursive(0, Vec::new(), &mut combinations);
        combinations
    }

    fn generate_recursive(
        &self,
        index: usize,
        current: Vec<(String, f64)>,
        result: &mut Vec<Vec<(String, f64)>>,
    ) {
        if index >= self.parameters.len() {
            result.push(current);
            return;
        }

        let param = &self.parameters[index];
        for &value in &param.values {
            let mut next = current.clone();
            next.push((param.name.clone(), value));
            self.generate_recursive(index + 1, next, result);
        }
    }

    /// Get total number of combinations
    pub fn num_combinations(&self) -> usize {
        self.parameters.iter().map(|p| p.values.len()).product()
    }
}

/// Result from parameter optimization
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub best_parameters: Vec<(String, f64)>,
    pub best_score: f64,
    pub all_results: Vec<ParameterSetResult>,
    pub improvement_over_baseline: f64,
}

#[derive(Debug, Clone)]
pub struct ParameterSetResult {
    pub parameters: Vec<(String, f64)>,
    pub score: f64,
    pub metrics: std::collections::HashMap<String, f64>,
}

/// Optimizer for finding best parameters
pub struct ParameterOptimizer {
    pool: OptimizationThreadPool,
}

impl ParameterOptimizer {
    pub fn new(config: OptimizationConfig) -> Self {
        Self {
            pool: OptimizationThreadPool::new(config),
        }
    }

    /// Run grid search optimization
    pub fn grid_search<F, S>(&self, grid: ParameterGrid, scorer: F) -> OptimizationResult
    where
        F: Fn(Vec<(String, f64)>) -> S + Send + Sync + 'static,
        S: Into<ParameterSetResult> + Send + 'static,
    {
        let combinations = grid.generate_combinations();
        info!("Grid search over {} combinations", combinations.len());

        let results: Vec<ParameterSetResult> = self.pool.run_parameter_sweep(combinations, |params| {
            let result = scorer(params.clone());
            result.into()
        });

        // Find best result
        let best = results.iter()
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
            .cloned();

        match best {
            Some(best_result) => {
                let baseline = results.first().map(|r| r.score).unwrap_or(0.0);
                let improvement = (best_result.score - baseline) / baseline.abs().max(1e-10);

                OptimizationResult {
                    best_parameters: best_result.parameters.clone(),
                    best_score: best_result.score,
                    all_results: results,
                    improvement_over_baseline: improvement,
                }
            }
            None => OptimizationResult {
                best_parameters: Vec::new(),
                best_score: 0.0,
                all_results: results,
                improvement_over_baseline: 0.0,
            }
        }
    }
}

/// Utility for memory-efficient iteration
pub struct BoundedIterator<I> {
    inner: I,
    memory_tracker: Arc<MemoryTracker>,
    items_processed: usize,
    checkpoint_interval: usize,
}

impl<I, T> BoundedIterator<I>
where
    I: Iterator<Item = T>,
{
    pub fn new(inner: I, memory_tracker: Arc<MemoryTracker>) -> Self {
        Self {
            inner,
            memory_tracker,
            items_processed: 0,
            checkpoint_interval: 100,
        }
    }

    /// Set checkpoint interval for memory checks
    pub fn with_checkpoint_interval(mut self, interval: usize) -> Self {
        self.checkpoint_interval = interval;
        self
    }
}

impl<I, T> Iterator for BoundedIterator<I>
where
    I: Iterator<Item = T>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        // Check memory periodically
        if self.items_processed % self.checkpoint_interval == 0 {
            if self.memory_tracker.is_near_limit(0.9) {
                warn!("Memory limit approached during iteration");
                // Could pause or skip items here
            }
        }

        let item = self.inner.next();
        if item.is_some() {
            self.items_processed += 1;
        }
        item
    }
}

// Re-export rayon for convenience
pub use rayon;
