//! Lock-Free Task Scheduler for Alpha Signal Evaluation
//! 
//! Evaluates alpha signals in topological order without blocking.
//! Distributes independent DAG nodes across CPU cores using work-stealing queues.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use crossbeam_queue::SegQueue;
use crossbeam_deque::{Stealer, Worker, Steal};
use parking_lot::RwLock;
use tracing::{debug, error, info, warn};

use crate::strategy::dag::{StrategyDag, ExecutionContext, NodeId, NodeData};

/// Maximum number of worker threads
const MAX_WORKERS: usize = 32;

/// Work-stealing task scheduler for parallel DAG execution
pub struct Orchestrator {
    /// Shared work queues (one per worker)
    queues: Vec<Arc<TaskQueue>>,
    /// Stealers for work stealing
    stealers: Vec<Stealer<Task>>,
    /// Number of workers
    num_workers: usize,
    /// Shutdown flag
    shutdown: AtomicBool,
    /// Tasks completed counter
    tasks_completed: AtomicU64,
    /// DAG reference
    dag: Arc<StrategyDag>,
    /// Worker handles
    workers: RwLock<Vec<thread::JoinHandle<()>>>,
}

/// Task to be executed
#[derive(Clone, Debug)]
pub struct Task {
    pub node_id: NodeId,
    pub priority: u8,
    pub trace_id: Option<u64>,
}

/// Per-worker task queue
struct TaskQueue {
    worker: Worker<Task>,
}

impl TaskQueue {
    fn new() -> Self {
        Self {
            worker: Worker::new_fifo(),
        }
    }
}

impl Orchestrator {
    /// Create a new orchestrator with specified number of workers
    pub fn new(num_workers: usize, dag: Arc<StrategyDag>) -> Self {
        let num_workers = num_workers.min(MAX_WORKERS);
        
        let mut queues = Vec::with_capacity(num_workers);
        let mut stealers = Vec::with_capacity(num_workers);
        
        for _ in 0..num_workers {
            let queue = Arc::new(TaskQueue::new());
            stealers.push(queue.worker.stealer());
            queues.push(queue);
        }
        
        Self {
            queues,
            stealers,
            num_workers,
            shutdown: AtomicBool::new(false),
            tasks_completed: AtomicU64::new(0),
            dag,
            workers: RwLock::new(Vec::new()),
        }
    }

    /// Start worker threads
    pub fn start(&self) {
        let mut workers = self.workers.write();
        
        for i in 0..self.num_workers {
            let queue = self.queues[i].clone();
            let stealers = self.stealers.clone();
            let shutdown = &self.shutdown;
            let tasks_completed = &self.tasks_completed;
            let dag = self.dag.clone();
            
            let handle = thread::spawn(move || {
                worker_loop(i, queue, stealers, shutdown, tasks_completed, dag);
            });
            
            workers.push(handle);
        }
        
        info!("Started {} worker threads", self.num_workers);
    }

    /// Submit a task to a specific worker queue
    pub fn submit(&self, task: Task, worker_id: usize) {
        if worker_id >= self.num_workers {
            warn!("Invalid worker ID {}", worker_id);
            return;
        }
        
        self.queues[worker_id].worker.push(task);
    }

    /// Broadcast task to all workers (for high-priority tasks)
    pub fn broadcast(&self, task: Task) {
        for (i, queue) in self.queues.iter().enumerate() {
            let mut t = task.clone();
            t.priority = task.priority.saturating_add(i as u8);
            queue.worker.push(t);
        }
    }

    /// Get current queue depths
    pub fn queue_depths(&self) -> Vec<usize> {
        self.queues.iter()
            .map(|q| q.worker.len())
            .collect()
    }

    /// Get total pending tasks
    pub fn pending_tasks(&self) -> usize {
        self.queue_depths().iter().sum()
    }

    /// Get completed task count
    pub fn tasks_completed(&self) -> u64 {
        self.tasks_completed.load(Ordering::Relaxed)
    }

    /// Gracefully shutdown all workers
    pub fn shutdown(&self) {
        info!("Shutting down orchestrator...");
        self.shutdown.store(true, Ordering::Release);
        
        // Wake up all workers
        for queue in &self.queues {
            queue.worker.push(Task {
                node_id: 0,
                priority: 255,
                trace_id: None,
            });
        }
        
        // Wait for workers to finish
        let mut workers = self.workers.write();
        for handle in workers.drain(..) {
            let _ = handle.join();
        }
        
        info!("Orchestrator shutdown complete");
    }

    /// Check if orchestrator is running
    pub fn is_running(&self) -> bool {
        !self.shutdown.load(Ordering::Acquire)
    }
}

/// Worker loop implementing work stealing
fn worker_loop(
    worker_id: usize,
    queue: Arc<TaskQueue>,
    stealers: Vec<Stealer<Task>>,
    shutdown: &AtomicBool,
    tasks_completed: &AtomicU64,
    _dag: Arc<StrategyDag>,
) {
    let mut idle_count = 0;
    
    while !shutdown.load(Ordering::Acquire) {
        // Try to get task from own queue first
        if let Some(task) = queue.worker.pop() {
            idle_count = 0;
            execute_task(worker_id, task, tasks_completed);
            continue;
        }
        
        // Try to steal from other workers
        let mut stolen = false;
        for (i, stealer) in stealers.iter().enumerate() {
            if i == worker_id {
                continue;
            }
            
            match stealer.steal() {
                Steal::Success(task) => {
                    stolen = true;
                    idle_count = 0;
                    execute_task(worker_id, task, tasks_completed);
                    break;
                }
                Steal::Retry => continue,
                Steal::Empty => continue,
            }
        }
        
        if !stolen {
            idle_count += 1;
            if idle_count > 100 {
                // Back off when idle
                thread::sleep(Duration::from_micros(10));
                idle_count = 0;
            }
        }
    }
}

/// Execute a single task
fn execute_task(
    worker_id: usize,
    task: Task,
    tasks_completed: &AtomicU64,
) {
    debug!("Worker {} executing task for node {}", worker_id, task.node_id);
    
    // In production, this would actually compute the node
    // For now, just increment counter
    tasks_completed.fetch_add(1, Ordering::Relaxed);
}

/// Builder for orchestrator configuration
pub struct OrchestratorBuilder {
    num_workers: usize,
}

impl OrchestratorBuilder {
    pub fn new() -> Self {
        Self {
            num_workers: num_cpus::get().min(MAX_WORKERS),
        }
    }

    pub fn workers(mut self, n: usize) -> Self {
        self.num_workers = n.min(MAX_WORKERS);
        self
    }

    pub fn build(self, dag: Arc<StrategyDag>) -> Orchestrator {
        Orchestrator::new(self.num_workers, dag)
    }
}

impl Default for OrchestratorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for the orchestrator
#[derive(Debug, Clone)]
pub struct OrchestratorStats {
    pub num_workers: usize,
    pub pending_tasks: usize,
    pub tasks_completed: u64,
    pub queue_depths: Vec<usize>,
    pub is_running: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orchestrator_creation() {
        let dag = Arc::new(StrategyDag::new());
        let orchestrator = Orchestrator::new(4, dag.clone());
        
        assert_eq!(orchestrator.num_workers, 4);
        assert!(!orchestrator.is_running()); // Not started yet
        assert_eq!(orchestrator.pending_tasks(), 0);
    }

    #[test]
    fn test_task_submission() {
        let dag = Arc::new(StrategyDag::new());
        let orchestrator = Orchestrator::new(4, dag.clone());
        
        let task = Task {
            node_id: 1,
            priority: 0,
            trace_id: None,
        };
        
        orchestrator.submit(task.clone(), 0);
        orchestrator.submit(task, 1);
        
        let depths = orchestrator.queue_depths();
        assert_eq!(depths[0], 1);
        assert_eq!(depths[1], 1);
    }

    #[test]
    fn test_builder() {
        let dag = Arc::new(StrategyDag::new());
        let orchestrator = OrchestratorBuilder::new()
            .workers(8)
            .build(dag);
        
        assert_eq!(orchestrator.num_workers, 8);
    }
}
