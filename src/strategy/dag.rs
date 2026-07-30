//! Directed Acyclic Graph (DAG) Execution Engine for Strategy Dependencies
//! 
//! Models alpha signal generation as a graph where nodes are calculations
//! and edges are data dependencies. Enables parallel execution of independent nodes.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::Instant;
use crossbeam_queue::SegQueue;
use parking_lot::RwLock;
use tracing::{debug, error, info, warn};

/// Maximum number of nodes in the DAG (bounded for RAM limits)
const MAX_DAG_NODES: usize = 1024;

/// Unique identifier for DAG nodes
pub type NodeId = u64;

/// Data type passed between nodes (zero-copy where possible)
#[derive(Clone, Debug)]
pub enum NodeData {
    F64(f64),
    F64Array([f64; 8]),
    I64(i64),
    Bool(bool),
    Bytes(Arc<[u8]>),
}

impl NodeData {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            NodeData::F64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            NodeData::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

/// Type of computation performed by a node
#[derive(Clone, Debug)]
pub enum NodeType {
    /// Input node (market data source)
    Input,
    /// Mathematical operation
    Math(MathOperation),
    /// Statistical calculation
    Statistics(StatOperation),
    /// Signal generation
    Signal(SignalType),
    /// Aggregation of multiple inputs
    Aggregate(AggregateType),
    /// Output/sink node
    Output,
}

#[derive(Clone, Debug)]
pub enum MathOperation {
    Add,
    Subtract,
    Multiply,
    Divide,
    Sqrt,
    Log,
    Exp,
    Abs,
    Negate,
}

#[derive(Clone, Debug)]
pub enum StatOperation {
    Mean,
    StdDev,
    Variance,
    Correlation,
    Covariance,
    ZScore,
}

#[derive(Clone, Debug)]
pub enum SignalType {
    ThresholdCross { threshold: f64, direction: Direction },
    MovingAverageCross,
    RsiOverbought { level: f64 },
    RsiOversold { level: f64 },
}

#[derive(Clone, Debug)]
pub enum Direction {
    Above,
    Below,
}

#[derive(Clone, Debug)]
pub enum AggregateType {
    Sum,
    Max,
    Min,
    Average,
    Concat,
}

/// A node in the execution DAG
pub struct DagNode {
    pub id: NodeId,
    pub name: String,
    pub node_type: NodeType,
    /// Input edge node IDs
    pub inputs: Vec<NodeId>,
    /// Output edge node IDs
    pub outputs: Vec<NodeId>,
    /// Cached computation result
    pub cached_result: Option<NodeData>,
    /// Whether this node has been computed in current cycle
    pub computed: bool,
    /// In-degree for topological sort
    pub in_degree: usize,
}

impl DagNode {
    pub fn new(id: NodeId, name: impl Into<String>, node_type: NodeType) -> Self {
        Self {
            id,
            name: name.into(),
            node_type,
            inputs: Vec::new(),
            outputs: Vec::new(),
            cached_result: None,
            computed: false,
            in_degree: 0,
        }
    }
}

/// Execution context for a single DAG evaluation cycle
pub struct ExecutionContext {
    /// Input data for this cycle
    pub inputs: HashMap<NodeId, NodeData>,
    /// Results from this cycle
    pub results: HashMap<NodeId, NodeData>,
    /// Cycle timestamp
    pub timestamp: Instant,
    /// Trace ID for observability
    pub trace_id: Option<u64>,
}

impl ExecutionContext {
    pub fn new() -> Self {
        Self {
            inputs: HashMap::new(),
            results: HashMap::new(),
            timestamp: Instant::now(),
            trace_id: None,
        }
    }

    pub fn with_trace_id(mut self, trace_id: u64) -> Self {
        self.trace_id = Some(trace_id);
        self
    }
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of node computation
#[derive(Debug, Clone)]
pub enum ComputeResult {
    Success(NodeData),
    Pending,
    Error(String),
}

/// DAG execution engine for strategy orchestration
pub struct StrategyDag {
    /// All nodes in the DAG
    nodes: RwLock<HashMap<NodeId, DagNode>>,
    /// Entry points (nodes with no inputs)
    entry_nodes: RwLock<Vec<NodeId>>,
    /// Exit points (nodes with no outputs)
    exit_nodes: RwLock<Vec<NodeId>>,
    /// Next available node ID
    next_node_id: AtomicU64,
    /// Execution statistics
    stats: DagStats,
    /// Pre-computed topological order
    topo_order: RwLock<Vec<NodeId>>,
    /// Dirty flag for topo recompute
    dirty: AtomicBool,
}

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

impl StrategyDag {
    /// Create a new empty DAG
    pub fn new() -> Self {
        Self {
            nodes: RwLock::new(HashMap::with_capacity(64)),
            entry_nodes: RwLock::new(Vec::new()),
            exit_nodes: RwLock::new(Vec::new()),
            next_node_id: AtomicU64::new(1),
            stats: DagStats::default(),
            topo_order: RwLock::new(Vec::new()),
            dirty: AtomicBool::new(true),
        }
    }

    /// Add a new node to the DAG
    pub fn add_node(&self, name: impl Into<String>, node_type: NodeType) -> Option<NodeId> {
        let mut nodes = self.nodes.write();
        
        if nodes.len() >= MAX_DAG_NODES {
            warn!("Maximum DAG nodes reached");
            return None;
        }

        let id = self.next_node_id.fetch_add(1, Ordering::Relaxed);
        let node = DagNode::new(id, name, node_type);
        
        nodes.insert(id, node);
        self.dirty.store(true, Ordering::Release);
        
        debug!("Added node {} to DAG", id);
        Some(id)
    }

    /// Add an edge between two nodes
    pub fn add_edge(&self, from_id: NodeId, to_id: NodeId) -> bool {
        let mut nodes = self.nodes.write();
        
        let from_node = nodes.get_mut(&from_id)?;
        let to_node = nodes.get_mut(&to_id)?;
        
        // Check for cycles before adding edge
        if self.would_create_cycle(from_id, to_id, &*nodes) {
            error!("Adding edge {} -> {} would create a cycle", from_id, to_id);
            return false;
        }
        
        from_node.outputs.push(to_id);
        to_node.inputs.push(from_id);
        to_node.in_degree += 1;
        
        self.dirty.store(true, Ordering::Release);
        debug!("Added edge {} -> {}", from_id, to_id);
        true
    }

    /// Check if adding an edge would create a cycle using DFS
    fn would_create_cycle(&self, from: NodeId, to: NodeId, nodes: &HashMap<NodeId, DagNode>) -> bool {
        if from == to {
            return true;
        }
        
        // DFS from 'to' to see if we can reach 'from'
        let mut visited = HashSet::new();
        let mut stack = vec![to];
        
        while let Some(current) = stack.pop() {
            if current == from {
                return true;
            }
            
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current);
            
            if let Some(node) = nodes.get(&current) {
                for &output in &node.outputs {
                    stack.push(output);
                }
            }
        }
        
        false
    }

    /// Remove a node from the DAG
    pub fn remove_node(&self, id: NodeId) -> bool {
        let mut nodes = self.nodes.write();
        if nodes.remove(&id).is_some() {
            self.dirty.store(true, Ordering::Release);
            true
        } else {
            false
        }
    }

    /// Compute topological order using Kahn's algorithm
    pub fn compute_topo_order(&self) -> Vec<NodeId> {
        let nodes = self.nodes.read();
        
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        let mut queue: VecDeque<NodeId> = VecDeque::new();
        let mut result: Vec<NodeId> = Vec::new();
        
        // Initialize in-degrees
        for (&id, node) in nodes.iter() {
            in_degree.insert(id, node.in_degree);
            if node.in_degree == 0 {
                queue.push_back(id);
            }
        }
        
        // Process nodes
        while let Some(node_id) = queue.pop_front() {
            result.push(node_id);
            
            if let Some(node) = nodes.get(&node_id) {
                for &output in &node.outputs {
                    let deg = in_degree.get_mut(&output).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(output);
                    }
                }
            }
        }
        
        if result.len() != nodes.len() {
            error!("DAG contains a cycle! Topological sort failed.");
            return Vec::new();
        }
        
        result
    }

    /// Execute the DAG with given inputs
    pub fn execute(&self, mut ctx: ExecutionContext) -> HashMap<NodeId, NodeData> {
        let start = Instant::now();
        
        // Ensure topo order is fresh
        if self.dirty.load(Ordering::Acquire) {
            let order = self.compute_topo_order();
            *self.topo_order.write() = order;
            self.dirty.store(false, Ordering::Release);
        }
        
        let topo_order = self.topo_order.read().clone();
        let nodes = self.nodes.read();
        
        // Initialize input nodes
        for (node_id, data) in ctx.inputs.drain() {
            ctx.results.insert(node_id, data);
        }
        
        // Execute nodes in topological order
        for node_id in topo_order {
            if let Some(node) = nodes.get(&node_id) {
                // Skip if all inputs aren't ready
                let inputs_ready = node.inputs.iter().all(|id| ctx.results.contains_key(id));
                
                if !inputs_ready && node.node_type != NodeType::Input {
                    continue;
                }
                
                // Gather input data
                let input_data: Vec<Option<&NodeData>> = node.inputs
                    .iter()
                    .map(|id| ctx.results.get(id))
                    .collect();
                
                // Compute node result
                if let Some(result) = self.compute_node(node, &input_data) {
                    match result {
                        ComputeResult::Success(data) => {
                            ctx.results.insert(node_id, data);
                        }
                        ComputeResult::Error(e) => {
                            error!("Node {} computation error: {}", node.name, e);
                        }
                        ComputeResult::Pending => {}
                    }
                }
            }
        }
        
        let elapsed = start.elapsed();
        self.stats.record_execution(elapsed, ctx.results.len());
        
        ctx.results
    }

    /// Compute a single node's result
    fn compute_node(&self, node: &DagNode, inputs: &[Option<&NodeData>]) -> Option<ComputeResult> {
        match &node.node_type {
            NodeType::Input => {
                // Input nodes should already have data in context
                None
            }
            NodeType::Math(op) => {
                self.compute_math(op, inputs)
            }
            NodeType::Statistics(op) => {
                self.compute_statistics(op, inputs)
            }
            NodeType::Signal(sig) => {
                self.compute_signal(sig, inputs)
            }
            NodeType::Aggregate(agg) => {
                self.compute_aggregate(agg, inputs)
            }
            NodeType::Output => {
                // Pass through first input
                inputs.first().and_then(|i| i.cloned()).map(ComputeResult::Success)
            }
        }
    }

    fn compute_math(&self, op: &MathOperation, inputs: &[Option<&NodeData>]) -> Option<ComputeResult> {
        match op {
            MathOperation::Add => {
                let a = inputs.get(0)?.and_then(|i| i.as_f64())?;
                let b = inputs.get(1)?.and_then(|i| i.as_f64())?;
                Some(ComputeResult::Success(NodeData::F64(a + b)))
            }
            MathOperation::Subtract => {
                let a = inputs.get(0)?.and_then(|i| i.as_f64())?;
                let b = inputs.get(1)?.and_then(|i| i.as_f64())?;
                Some(ComputeResult::Success(NodeData::F64(a - b)))
            }
            MathOperation::Multiply => {
                let a = inputs.get(0)?.and_then(|i| i.as_f64())?;
                let b = inputs.get(1)?.and_then(|i| i.as_f64())?;
                Some(ComputeResult::Success(NodeData::F64(a * b)))
            }
            MathOperation::Divide => {
                let a = inputs.get(0)?.and_then(|i| i.as_f64())?;
                let b = inputs.get(1)?.and_then(|i| i.as_f64())?;
                if b.abs() < 1e-10 {
                    return Some(ComputeResult::Error("Division by zero".into()));
                }
                Some(ComputeResult::Success(NodeData::F64(a / b)))
            }
            MathOperation::Sqrt => {
                let a = inputs.get(0)?.and_then(|i| i.as_f64())?;
                Some(ComputeResult::Success(NodeData::F64(a.sqrt())))
            }
            MathOperation::Log => {
                let a = inputs.get(0)?.and_then(|i| i.as_f64())?;
                Some(ComputeResult::Success(NodeData::F64(a.ln())))
            }
            MathOperation::Exp => {
                let a = inputs.get(0)?.and_then(|i| i.as_f64())?;
                Some(ComputeResult::Success(NodeData::F64(a.exp())))
            }
            MathOperation::Abs => {
                let a = inputs.get(0)?.and_then(|i| i.as_f64())?;
                Some(ComputeResult::Success(NodeData::F64(a.abs())))
            }
            MathOperation::Negate => {
                let a = inputs.get(0)?.and_then(|i| i.as_f64())?;
                Some(ComputeResult::Success(NodeData::F64(-a)))
            }
        }
    }

    fn compute_statistics(&self, _op: &StatOperation, _inputs: &[Option<&NodeData>]) -> Option<ComputeResult> {
        // Placeholder for statistical operations
        Some(ComputeResult::Success(NodeData::F64(0.0)))
    }

    fn compute_signal(&self, sig: &SignalType, inputs: &[Option<&NodeData>]) -> Option<ComputeResult> {
        match sig {
            SignalType::ThresholdCross { threshold, direction } => {
                let value = inputs.get(0)?.and_then(|i| i.as_f64())?;
                let triggered = match direction {
                    Direction::Above => value > *threshold,
                    Direction::Below => value < *threshold,
                };
                Some(ComputeResult::Success(NodeData::Bool(triggered)))
            }
            SignalType::RsiOverbought { level } => {
                let rsi = inputs.get(0)?.and_then(|i| i.as_f64())?;
                Some(ComputeResult::Success(NodeData::Bool(rsi > *level)))
            }
            SignalType::RsiOversold { level } => {
                let rsi = inputs.get(0)?.and_then(|i| i.as_f64())?;
                Some(ComputeResult::Success(NodeData::Bool(rsi < *level)))
            }
            _ => Some(ComputeResult::Success(NodeData::Bool(false))),
        }
    }

    fn compute_aggregate(&self, agg: &AggregateType, inputs: &[Option<&NodeData>]) -> Option<ComputeResult> {
        match agg {
            AggregateType::Sum => {
                let sum: f64 = inputs.iter()
                    .filter_map(|i| i.and_then(|d| d.as_f64()))
                    .sum();
                Some(ComputeResult::Success(NodeData::F64(sum)))
            }
            AggregateType::Max => {
                let max = inputs.iter()
                    .filter_map(|i| i.and_then(|d| d.as_f64()))
                    .fold(f64::NEG_INFINITY, f64::max);
                Some(ComputeResult::Success(NodeData::F64(max)))
            }
            AggregateType::Min => {
                let min = inputs.iter()
                    .filter_map(|i| i.and_then(|d| d.as_f64()))
                    .fold(f64::INFINITY, f64::min);
                Some(ComputeResult::Success(NodeData::F64(min)))
            }
            AggregateType::Average => {
                let values: Vec<f64> = inputs.iter()
                    .filter_map(|i| i.and_then(|d| d.as_f64()))
                    .collect();
                if values.is_empty() {
                    return Some(ComputeResult::Error("No values for average".into()));
                }
                let avg = values.iter().sum::<f64>() / values.len() as f64;
                Some(ComputeResult::Success(NodeData::F64(avg)))
            }
            _ => None,
        }
    }

    /// Get execution statistics
    pub fn stats(&self) -> DagStats {
        self.stats.clone()
    }

    /// Get node count
    pub fn node_count(&self) -> usize {
        self.nodes.read().len()
    }

    /// Clear all cached results
    pub fn clear_cache(&self) {
        let mut nodes = self.nodes.write();
        for node in nodes.values_mut() {
            node.cached_result = None;
            node.computed = false;
        }
    }
}

impl Default for StrategyDag {
    fn default() -> Self {
        Self::new()
    }
}

/// DAG execution statistics
#[derive(Default, Clone, Debug)]
pub struct DagStats {
    pub total_executions: u64,
    pub avg_execution_time_us: u64,
    pub max_execution_time_us: u64,
    pub min_execution_time_us: u64,
    pub last_execution_time_us: u64,
    samples: u64,
}

impl DagStats {
    pub fn record_execution(&mut self, duration: std::time::Duration, _nodes_computed: usize) {
        let us = duration.as_micros() as u64;
        
        self.total_executions += 1;
        self.last_execution_time_us = us;
        
        if self.samples == 0 {
            self.max_execution_time_us = us;
            self.min_execution_time_us = us;
        } else {
            self.max_execution_time_us = self.max_execution_time_us.max(us);
            self.min_execution_time_us = self.min_execution_time_us.min(us);
        }
        
        // Running average
        let total = self.avg_execution_time_us.saturating_mul(self.samples);
        self.samples = self.samples.saturating_add(1);
        self.avg_execution_time_us = total.saturating_add(us) / self.samples;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dag_creation() {
        let dag = StrategyDag::new();
        
        let input = dag.add_node("price_input", NodeType::Input);
        let sqrt_node = dag.add_node("sqrt_price", NodeType::Math(MathOperation::Sqrt));
        let output = dag.add_node("output", NodeType::Output);
        
        assert!(input.is_some());
        assert!(sqrt_node.is_some());
        assert!(output.is_some());
        
        assert!(dag.add_edge(input.unwrap(), sqrt_node.unwrap()));
        assert!(dag.add_edge(sqrt_node.unwrap(), output.unwrap()));
        
        assert_eq!(dag.node_count(), 3);
    }

    #[test]
    fn test_cycle_detection() {
        let dag = StrategyDag::new();
        
        let a = dag.add_node("a", NodeType::Input).unwrap();
        let b = dag.add_node("b", NodeType::Math(MathOperation::Add)).unwrap();
        let c = dag.add_node("c", NodeType::Output).unwrap();
        
        dag.add_edge(a, b);
        dag.add_edge(b, c);
        
        // This would create a cycle
        assert!(!dag.add_edge(c, a));
    }

    #[test]
    fn test_dag_execution() {
        let dag = StrategyDag::new();
        
        let input = dag.add_node("input", NodeType::Input).unwrap();
        let doubled = dag.add_node("double", NodeType::Math(MathOperation::Multiply)).unwrap();
        let output = dag.add_node("output", NodeType::Output).unwrap();
        
        dag.add_edge(input, doubled);
        dag.add_edge(doubled, output);
        
        let mut ctx = ExecutionContext::new();
        ctx.inputs.insert(input, NodeData::F64(5.0));
        ctx.inputs.insert(doubled, NodeData::F64(2.0)); // Second input for multiply
        
        let results = dag.execute(ctx);
        
        assert!(results.contains_key(&output));
    }
}
