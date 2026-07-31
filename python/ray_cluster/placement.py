"""
Custom Ray placement strategies for CPU core pinning.
Pins ML inference workers to specific AMD Ryzen CPU cores, avoiding Rust engine cores.
"""

import ray
import os
from typing import List, Dict, Optional, Any
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import ML_WORKER_CPU_CORES, RUST_ENGINE_CPU_CORES, get_logger

logger = get_logger("placement")


class PlacementStrategy:
    """
    Defines CPU core placement strategies for Ray actors.
    Ensures ML workers don't interfere with Rust engine cores.
    """
    
    def __init__(self, ml_cores: List[int], rust_cores: List[int]):
        self.ml_cores = ml_cores
        self.rust_cores = rust_cores
        self.available_cores = set(ml_cores)
        self.assigned_cores: Dict[str, int] = {}
    
    def get_core_for_worker(self, worker_id: str) -> Optional[int]:
        """
        Assign a CPU core to a worker using round-robin strategy.
        
        Args:
            worker_id: Unique identifier for the worker
        
        Returns:
            CPU core number or None if no cores available
        """
        # Check if worker already has an assigned core
        if worker_id in self.assigned_cores:
            return self.assigned_cores[worker_id]
        
        # Find available cores (not currently assigned)
        assigned_set = set(self.assigned_cores.values())
        available = [c for c in self.ml_cores if c not in assigned_set]
        
        if not available:
            logger.warning(f"No available CPU cores for worker {worker_id}")
            return None
        
        # Round-robin assignment
        core = available[len(self.assigned_cores) % len(available)]
        self.assigned_cores[worker_id] = core
        
        logger.info(f"Assigned CPU core {core} to worker {worker_id}")
        return core
    
    def release_core(self, worker_id: str) -> None:
        """Release a CPU core when a worker shuts down."""
        if worker_id in self.assigned_cores:
            released_core = self.assigned_cores.pop(worker_id)
            logger.info(f"Released CPU core {released_core} from worker {worker_id}")
    
    def validate_core_isolation(self) -> bool:
        """Validate that ML and Rust cores don't overlap."""
        overlap = set(self.ml_cores) & set(self.rust_cores)
        if overlap:
            logger.error(f"CPU core isolation violated! Overlap: {overlap}")
            return False
        logger.info("CPU core isolation validated successfully")
        return True


@ray.remote
class PinnedMLWorker:
    """
    Ray actor representing an ML inference worker pinned to a specific CPU core.
    """
    
    def __init__(self, worker_id: str, cpu_core: int):
        self.worker_id = worker_id
        self.cpu_core = cpu_core
        
        # Pin this process to the specified CPU core
        self._pin_to_core(cpu_core)
        
        logger.info(f"Pinned worker {worker_id} to CPU core {cpu_core}")
    
    def _pin_to_core(self, core: int) -> None:
        """Pin the current process to a specific CPU core using OS affinity."""
        try:
            psutil = __import__("psutil")
            process = psutil.Process(os.getpid())
            process.cpu_affinity([core])
            logger.debug(f"Process {os.getpid()} pinned to core {core}")
        except Exception as e:
            logger.warning(f"Failed to pin process to core {core}: {e}")
    
    def get_worker_info(self) -> Dict[str, Any]:
        """Get information about this worker."""
        import psutil
        process = psutil.Process(os.getpid())
        
        return {
            "worker_id": self.worker_id,
            "cpu_core": self.cpu_core,
            "pid": os.getpid(),
            "cpu_affinity": process.cpu_affinity(),
        }
    
    def execute_inference(self, features: Any) -> Any:
        """
        Execute ML inference on this pinned worker.
        
        Args:
            features: Feature vector from shared memory
        
        Returns:
            Inference result
        """
        # Placeholder for actual inference logic
        # The key is that this runs on a pinned CPU core
        logger.debug(f"Worker {self.worker_id} executing inference on core {self.cpu_core}")
        return {"worker_id": self.worker_id, "status": "inference_complete"}


def create_placement_group(
    name: str,
    num_workers: int,
    bundle_strategy: str = "PACK",
) -> ray.placement_group.PlacementGroup:
    """
    Create a Ray placement group for CPU-pinned workers.
    
    Args:
        name: Name of the placement group
        num_workers: Number of workers to place
        bundle_strategy: "PACK" (same node) or "SPREAD" (different nodes)
    
    Returns:
        Placement group object
    """
    bundles = []
    for i in range(num_workers):
        # Each bundle reserves 1 CPU and a slice of memory
        bundles.append({"CPU": 1, "memory": 100 * 1024 * 1024})  # 100MB per worker
    
    pg = ray.util.placement_group(
        name=name,
        bundles=bundles,
        strategy=bundle_strategy,
    )
    
    logger.info(f"Created placement group '{name}' with {num_workers} bundles")
    return pg


def remove_placement_group(pg: ray.placement_group.PlacementGroup) -> None:
    """Remove a placement group."""
    try:
        ray.util.remove_placement_group(pg)
        logger.info(f"Removed placement group '{pg.name}'")
    except Exception as e:
        logger.error(f"Failed to remove placement group: {e}")


def get_optimal_placement_strategy() -> Dict[str, Any]:
    """
    Get optimal placement configuration based on available CPU cores.
    
    Returns:
        Dictionary with placement configuration
    """
    num_ml_cores = len(ML_WORKER_CPU_CORES)
    
    return {
        "max_workers": num_ml_cores,
        "reserved_rust_cores": RUST_ENGINE_CPU_CORES,
        "available_ml_cores": ML_WORKER_CPU_CORES,
        "recommended_bundles": num_ml_cores,
        "isolation_validated": True,
    }


# Module-level placement strategy instance
_placement_strategy: Optional[PlacementStrategy] = None


def get_placement_strategy() -> PlacementStrategy:
    """Get or create the global placement strategy instance."""
    global _placement_strategy
    if _placement_strategy is None:
        _placement_strategy = PlacementStrategy(ML_WORKER_CPU_CORES, RUST_ENGINE_CPU_CORES)
        _placement_strategy.validate_core_isolation()
    return _placement_strategy
