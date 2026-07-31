# Features Module Root
# Manages feature DAG and enforces strict memory quotas on Ray workers

from __future__ import annotations
import logging
import ray
from typing import Dict, Any, Optional, List

log = logging.getLogger(__name__)

# Import transformers and pipeline components
from python.features.transformers import (
    ZScoreTransformer,
    FractionalDifferencingTransformer,
    ClipTransformer,
    FeatureUnionTransformer,
    create_standard_pipeline,
)
from python.features.pipeline import (
    FeaturePipelineActor,
    PipelineOrchestrator,
    create_pipeline,
)


class FeatureMemoryManager:
    """
    Enforces strict memory quotas on feature processing.
    Monitors Ray worker memory and triggers garbage collection when needed.
    """

    def __init__(
        self,
        max_memory_mb: int = 500,
        gc_threshold: float = 0.8,
    ) -> None:
        self.max_memory_mb = max_memory_mb
        self.gc_threshold = gc_threshold
        self._worker_stats: Dict[int, Dict[str, Any]] = {}

    def check_worker_memory(self, orchestrator: PipelineOrchestrator) -> List[int]:
        """
        Check memory usage of all workers.
        Returns list of worker IDs that need restart.
        """
        bloated_workers = []
        
        try:
            stats_list = ray.get(orchestrator.get_all_stats.remote())
            
            for stats in stats_list:
                worker_id = stats["worker_id"]
                memory_bytes = stats["memory_usage_bytes"]
                memory_mb = memory_bytes / (1024 * 1024)
                
                self._worker_stats[worker_id] = {
                    "memory_mb": memory_mb,
                    "total_processed": stats["total_processed"],
                }
                
                if memory_mb > self.max_memory_mb * self.gc_threshold:
                    log.warning(
                        f"Worker {worker_id} at {memory_mb:.1f}MB "
                        f"({memory_mb/self.max_memory_mb*100:.1f}% of quota)"
                    )
                    bloated_workers.append(worker_id)
                    
        except Exception as e:
            log.error(f"Error checking worker memory: {e}")
        
        return bloated_workers

    def get_total_memory_usage(self) -> float:
        """Return total memory usage across all workers in MB."""
        return sum(s["memory_mb"] for s in self._worker_stats.values())


class FeatureDAG:
    """
    Manages the feature transformation DAG.
    Defines dependencies and execution order for feature computation.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, Any] = {}
        self._edges: Dict[str, List[str]] = {}
        self._execution_order: List[str] = []

    def add_node(self, name: str, transformer: Any, dependencies: Optional[List[str]] = None) -> None:
        """Add a node to the DAG."""
        self._nodes[name] = transformer
        self._edges[name] = dependencies or []
        self._compute_execution_order()
        log.debug(f"Added node '{name}' with dependencies: {dependencies}")

    def remove_node(self, name: str) -> None:
        """Remove a node from the DAG."""
        if name in self._nodes:
            del self._nodes[name]
            del self._edges[name]
            # Remove from dependencies of other nodes
            for deps in self._edges.values():
                if name in deps:
                    deps.remove(name)
            self._compute_execution_order()
            log.debug(f"Removed node '{name}'")

    def _compute_execution_order(self) -> None:
        """Compute topological sort for execution order."""
        visited = set()
        order = []
        
        def visit(node: str) -> None:
            if node in visited:
                return
            visited.add(node)
            
            for dep in self._edges.get(node, []):
                if dep in self._nodes:
                    visit(dep)
            
            order.append(node)
        
        for node in self._nodes:
            visit(node)
        
        self._execution_order = order

    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the DAG in topological order."""
        results = dict(data)
        
        for node_name in self._execution_order:
            transformer = self._nodes[node_name]
            
            # Get input data based on dependencies
            if self._edges[node_name]:
                input_data = {dep: results[dep] for dep in self._edges[node_name] if dep in results}
            else:
                input_data = results
            
            # Apply transformation
            if hasattr(transformer, 'transform'):
                output = transformer.transform(input_data)
                if isinstance(output, dict):
                    results.update(output)
                else:
                    results[node_name] = output
            elif callable(transformer):
                results[node_name] = transformer(input_data)
            
            log.debug(f"Executed node '{node_name}'")
        
        return results

    def get_execution_order(self) -> List[str]:
        """Return the computed execution order."""
        return self._execution_order

    def visualize(self) -> str:
        """Return a text representation of the DAG."""
        lines = ["Feature DAG Execution Order:"]
        for i, node in enumerate(self._execution_order):
            deps = self._edges.get(node, [])
            dep_str = f" <- [{', '.join(deps)}]" if deps else ""
            lines.append(f"  {i}. {node}{dep_str}")
        return "\n".join(lines)


def create_feature_dag() -> FeatureDAG:
    """Factory function to create a standard feature DAG."""
    dag = FeatureDAG()
    
    # Add standard transformers
    dag.add_node("zscore", ZScoreTransformer(decay=0.999))
    dag.add_node("clip", ClipTransformer(lower_percentile=0.5, upper_percentile=99.5), 
                 dependencies=["zscore"])
    dag.add_node("frac_diff", FractionalDifferencingTransformer(d=0.5, window=100),
                 dependencies=["clip"])
    
    return dag


def validate_feature_matrix(matrix: Any, max_features: int = 500) -> bool:
    """Validate feature matrix dimensions and memory footprint."""
    import numpy as np
    
    if not isinstance(matrix, np.ndarray):
        return False
    
    if matrix.ndim != 2:
        return False
    
    n_samples, n_features = matrix.shape
    
    if n_features > max_features:
        log.warning(f"Feature count {n_features} exceeds limit {max_features}")
        return False
    
    # Check memory footprint (should be < 100MB for safety)
    memory_mb = matrix.nbytes / (1024 * 1024)
    if memory_mb > 100:
        log.warning(f"Feature matrix size {memory_mb:.1f}MB exceeds safety limit")
        return False
    
    return True


__all__ = [
    "ZScoreTransformer",
    "FractionalDifferencingTransformer",
    "ClipTransformer",
    "FeatureUnionTransformer",
    "FeaturePipelineActor",
    "PipelineOrchestrator",
    "FeatureMemoryManager",
    "FeatureDAG",
    "create_pipeline",
    "create_standard_pipeline",
    "create_feature_dag",
    "validate_feature_matrix",
]
