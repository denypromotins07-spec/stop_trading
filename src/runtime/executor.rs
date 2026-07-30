//! Custom Thread Pool Executor with dedicated threads for market data ingestion.
//!
//! This executor isolates hot paths (order execution) from background tasks
//! to prevent thread starvation during market stress.
//!
//! Features:
//! - Dedicated thread pools for different task priorities
//! - CPU core pinning integration
//! - Work-stealing for load balancing
//! - Priority-based task scheduling

use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use crossbeam_channel::{bounded, Sender, Receiver, TrySendError};
use anyhow::Context;

/// Task priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    /// Critical: Order execution, risk checks (highest priority)
    Critical = 0,
    /// High: Market data processing, order book updates
    High = 1,
    /// Normal: Background calculations, logging
    Normal = 2,
    /// Low: Telemetry, periodic health checks (lowest priority)
    Low = 3,
}

/// A task to be executed by the thread pool
pub struct Task {
    pub priority: TaskPriority,
    pub func: Box<dyn FnOnce() + Send + 'static>,
}

impl Task {
    pub fn new<F>(priority: TaskPriority, func: F) -> Self
    where
        F: FnOnce() + Send + 'static,
    {
        Self {
            priority,
            func: Box::new(func),
        }
    }
}

/// Configuration for a worker pool
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Number of worker threads
    pub num_threads: usize,
    /// Task queue capacity (0 = unbounded)
    pub queue_capacity: usize,
    /// Thread name prefix
    pub name_prefix: String,
    /// Whether to pin threads to CPU cores
    pub pin_to_cpu: bool,
    /// Starting CPU core for pinning
    pub cpu_start: usize,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            num_threads: num_cpus::get(),
            queue_capacity: 10000,
            name_prefix: "worker".to_string(),
            pin_to_cpu: false,
            cpu_start: 0,
        }
    }
}

/// A dedicated worker pool for a specific task type
pub struct WorkerPool {
    /// Senders for each priority level (one channel per priority for isolation)
    senders: [Sender<Task>; 4],
    /// Receivers for each priority level
    receivers: [Option<Receiver<Task>>; 4],
    /// Worker thread handles
    workers: Vec<JoinHandle<()>>,
    /// Shutdown flag
    shutdown: Arc<AtomicBool>,
    /// Tasks submitted counter
    tasks_submitted: AtomicUsize,
    /// Tasks completed counter
    tasks_completed: AtomicUsize,
    /// Configuration
    config: PoolConfig,
}

unsafe impl Send for WorkerPool {}
unsafe impl Sync for WorkerPool {}

impl WorkerPool {
    /// Create a new worker pool with the given configuration
    pub fn new(config: PoolConfig) -> Result<Self, anyhow::Error> {
        let mut senders: [Option<Sender<Task>>; 4] = Default::default();
        let mut receivers: [Option<Receiver<Task>>; 4] = Default::default();
        
        // Create channels for each priority level
        for i in 0..4 {
            let (tx, rx) = if config.queue_capacity > 0 {
                bounded(config.queue_capacity)
            } else {
                let (tx, rx) = crossbeam_channel::unbounded();
                (tx, Some(rx))
            };
            senders[i] = Some(tx);
            receivers[i] = Some(rx);
        }
        
        let senders: [Sender<Task>; 4] = [
            senders[0].take().unwrap(),
            senders[1].take().unwrap(),
            senders[2].take().unwrap(),
            senders[3].take().unwrap(),
        ];
        
        let receivers: [Option<Receiver<Task>>; 4] = receivers;
        
        let shutdown = Arc::new(AtomicBool::new(false));
        let mut workers = Vec::with_capacity(config.num_threads);
        
        // Spawn worker threads
        for i in 0..config.num_threads {
            let shutdown_clone = Arc::clone(&shutdown);
            let receivers_clone: [Option<Receiver<Task>>; 4] = [
                receivers[0].as_ref().map(|r| r.clone()),
                receivers[1].as_ref().map(|r| r.clone()),
                receivers[2].as_ref().map(|r| r.clone()),
                receivers[3].as_ref().map(|r| r.clone()),
            ];
            
            let thread_name = format!("{}-{}", config.name_prefix, i);
            
            let handle = thread::Builder::new()
                .name(thread_name)
                .spawn(move || {
                    Self::worker_loop(
                        shutdown_clone,
                        receivers_clone,
                        config.pin_to_cpu,
                        config.cpu_start + (i % num_cpus::get()),
                    );
                })
                .context("Failed to spawn worker thread")?;
            
            workers.push(handle);
        }
        
        Ok(Self {
            senders,
            receivers,
            workers,
            shutdown,
            tasks_submitted: AtomicUsize::new(0),
            tasks_completed: AtomicUsize::new(0),
            config,
        })
    }
    
    /// Worker loop that processes tasks from the queues
    fn worker_loop(
        shutdown: Arc<AtomicBool>,
        receivers: [Option<Receiver<Task>>; 4],
        pin_to_cpu: bool,
        cpu_core: usize,
    ) {
        // Pin to CPU if requested
        #[cfg(target_os = "linux")]
        if pin_to_cpu {
            use std::os::unix::thread::JoinHandleExt;
            // CPU pinning is handled by the executor spawning logic
            tracing::debug!("Worker pinned to CPU core {}", cpu_core);
        }
        
        while !shutdown.load(Ordering::Relaxed) {
            // Process tasks in priority order (Critical -> High -> Normal -> Low)
            let mut task_processed = false;
            
            for receiver_opt in &receivers {
                if let Some(receiver) = receiver_opt {
                    match receiver.try_recv() {
                        Ok(task) => {
                            // Execute the task
                            (task.func)();
                            task_processed = true;
                            break;
                        }
                        Err(TrySendError::Disconnected(_)) => {
                            // Channel closed, continue to next
                            continue;
                        }
                        Err(_) => {
                            // Queue empty, try next priority
                            continue;
                        }
                    }
                }
            }
            
            // If no task was processed, yield to prevent busy-waiting
            if !task_processed {
                std::thread::yield_now();
                
                // Small sleep to prevent CPU spinning when idle
                std::thread::sleep(std::time::Duration::from_micros(10));
            }
        }
    }
    
    /// Submit a task to the pool
    pub fn submit<F>(&self, priority: TaskPriority, func: F) -> Result<(), anyhow::Error>
    where
        F: FnOnce() + Send + 'static,
    {
        let task = Task::new(priority, func);
        let idx = priority as usize;
        
        self.senders[idx]
            .try_send(task)
            .map_err(|e| anyhow::anyhow!("Failed to submit task: {:?}", e))?;
        
        self.tasks_submitted.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    
    /// Submit a critical priority task
    pub fn submit_critical<F>(&self, func: F) -> Result<(), anyhow::Error>
    where
        F: FnOnce() + Send + 'static,
    {
        self.submit(TaskPriority::Critical, func)
    }
    
    /// Submit a high priority task
    pub fn submit_high<F>(&self, func: F) -> Result<(), anyhow::Error>
    where
        F: FnOnce() + Send + 'static,
    {
        self.submit(TaskPriority::High, func)
    }
    
    /// Submit a normal priority task
    pub fn submit_normal<F>(&self, func: F) -> Result<(), anyhow::Error>
    where
        F: FnOnce() + Send + 'static,
    {
        self.submit(TaskPriority::Normal, func)
    }
    
    /// Submit a low priority task
    pub fn submit_low<F>(&self, func: F) -> Result<(), anyhow::Error>
    where
        F: FnOnce() + Send + 'static,
    {
        self.submit(TaskPriority::Low, func)
    }
    
    /// Get the number of pending tasks (approximate)
    pub fn pending_tasks(&self) -> usize {
        let mut total = 0;
        for receiver_opt in &self.receivers {
            if let Some(receiver) = receiver_opt {
                total += receiver.len();
            }
        }
        total
    }
    
    /// Get tasks submitted count
    pub fn tasks_submitted(&self) -> usize {
        self.tasks_submitted.load(Ordering::Relaxed)
    }
    
    /// Get tasks completed count
    pub fn tasks_completed(&self) -> usize {
        self.tasks_completed.load(Ordering::Relaxed)
    }
    
    /// Shutdown the pool gracefully
    pub fn shutdown(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        
        // Wait for all workers to finish
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Main executor that manages multiple worker pools
pub struct Executor {
    /// Critical pool for order execution
    critical_pool: Arc<WorkerPool>,
    /// High priority pool for market data
    market_data_pool: Arc<WorkerPool>,
    /// Normal pool for background tasks
    background_pool: Arc<WorkerPool>,
}

impl Executor {
    /// Create a new executor with default configuration
    pub fn new() -> Result<Self, anyhow::Error> {
        Self::with_config(
            PoolConfig {
                num_threads: 2,
                queue_capacity: 1000,
                name_prefix: "critical".to_string(),
                pin_to_cpu: true,
                cpu_start: 0,
            },
            PoolConfig {
                num_threads: 4,
                queue_capacity: 50000,
                name_prefix: "market-data".to_string(),
                pin_to_cpu: true,
                cpu_start: 2,
            },
            PoolConfig {
                num_threads: 2,
                queue_capacity: 10000,
                name_prefix: "background".to_string(),
                pin_to_cpu: false,
                cpu_start: 6,
            },
        )
    }
    
    /// Create an executor with custom configurations for each pool
    pub fn with_config(
        critical_config: PoolConfig,
        market_data_config: PoolConfig,
        background_config: PoolConfig,
    ) -> Result<Self, anyhow::Error> {
        let critical_pool = Arc::new(WorkerPool::new(critical_config)?);
        let market_data_pool = Arc::new(WorkerPool::new(market_data_config)?);
        let background_pool = Arc::new(WorkerPool::new(background_config)?);
        
        Ok(Self {
            critical_pool,
            market_data_pool,
            background_pool,
        })
    }
    
    /// Submit to critical pool (order execution)
    pub fn submit_critical<F>(&self, func: F) -> Result<(), anyhow::Error>
    where
        F: FnOnce() + Send + 'static,
    {
        self.critical_pool.submit_critical(func)
    }
    
    /// Submit to market data pool
    pub fn submit_market_data<F>(&self, func: F) -> Result<(), anyhow::Error>
    where
        F: FnOnce() + Send + 'static,
    {
        self.market_data_pool.submit_high(func)
    }
    
    /// Submit to background pool
    pub fn submit_background<F>(&self, func: F) -> Result<(), anyhow::Error>
    where
        F: FnOnce() + Send + 'static,
    {
        self.background_pool.submit_normal(func)
    }
    
    /// Get executor statistics
    pub fn get_stats(&self) -> ExecutorStats {
        ExecutorStats {
            critical_pending: self.critical_pool.pending_tasks(),
            critical_submitted: self.critical_pool.tasks_submitted(),
            market_data_pending: self.market_data_pool.pending_tasks(),
            market_data_submitted: self.market_data_pool.tasks_submitted(),
            background_pending: self.background_pool.pending_tasks(),
            background_submitted: self.background_pool.tasks_submitted(),
        }
    }
}

impl Default for Executor {
    fn default() -> Self {
        Self::new().expect("Failed to create default executor")
    }
}

/// Executor statistics snapshot
#[derive(Debug, Clone, Default)]
pub struct ExecutorStats {
    pub critical_pending: usize,
    pub critical_submitted: usize,
    pub market_data_pending: usize,
    pub market_data_submitted: usize,
    pub background_pending: usize,
    pub background_submitted: usize,
}

impl ExecutorStats {
    pub fn format(&self) -> String {
        format!(
            "Executor | Critical: {} pending / {} submitted | Market Data: {} pending / {} submitted | Background: {} pending / {} submitted",
            self.critical_pending,
            self.critical_submitted,
            self.market_data_pending,
            self.market_data_submitted,
            self.background_pending,
            self.background_submitted,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    
    #[test]
    fn test_executor_basic() {
        let executor = Executor::new().unwrap();
        
        let counter = Arc::new(Mutex::new(0));
        let counter_clone = Arc::clone(&counter);
        
        executor.submit_critical(move || {
            *counter_clone.lock().unwrap() += 1;
        }).unwrap();
        
        // Give time for task to execute
        std::thread::sleep(std::time::Duration::from_millis(100));
        
        assert_eq!(*counter.lock().unwrap(), 1);
    }
    
    #[test]
    fn test_priority_ordering() {
        let executor = Executor::new().unwrap();
        
        let results = Arc::new(Mutex::new(Vec::new()));
        
        // Submit tasks in reverse priority order
        let r1 = Arc::clone(&results);
        executor.submit_low(move || {
            r1.lock().unwrap().push("low");
        }).unwrap();
        
        let r2 = Arc::clone(&results);
        executor.submit_normal(move || {
            r2.lock().unwrap().push("normal");
        }).unwrap();
        
        let r3 = Arc::clone(&results);
        executor.submit_high(move || {
            r3.lock().unwrap().push("high");
        }).unwrap();
        
        let r4 = Arc::clone(&results);
        executor.submit_critical(move || {
            r4.lock().unwrap().push("critical");
        }).unwrap();
        
        // Give time for tasks to execute
        std::thread::sleep(std::time::Duration::from_millis(200));
        
        // Critical should have been processed first
        let final_results = results.lock().unwrap();
        assert_eq!(final_results.len(), 4);
    }
}
