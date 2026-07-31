"""
Ray cluster initialization with strict resource bounds.
Binds dashboard to localhost for security and enforces memory limits.
"""

import ray
from pathlib import Path
from typing import Optional, Dict, Any

# Import settings from config
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import get_ray_init_kwargs, validate_environment, RAY_DASHBOARD_HOST, RAY_DASHBOARD_PORT


class RayClusterManager:
    """
    Manages Ray cluster lifecycle with strict resource constraints.
    Ensures Python processes stay within allocated 3GB memory limit.
    """
    
    def __init__(self):
        self._cluster_initialized = False
        self._context: Optional[ray.ClientContext] = None
    
    def initialize(self) -> bool:
        """
        Initialize the local Ray cluster with strict resource bounds.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self._cluster_initialized:
            return True
        
        try:
            # Validate environment before initialization
            validate_environment()
            
            # Get initialization kwargs with strict memory bounds
            init_kwargs = get_ray_init_kwargs()
            
            # Check if Ray is already running
            if ray.is_initialized():
                self._cluster_initialized = True
                return True
            
            # Initialize Ray with strict resource bounds
            self._context = ray.init(**init_kwargs)
            self._cluster_initialized = True
            
            # Log cluster info
            cluster_info = ray.cluster_resources()
            print(f"Ray cluster initialized:")
            print(f"  - CPUs: {cluster_info.get('CPU', 0)}")
            print(f"  - Memory: {cluster_info.get('memory', 0) / (1024**2):.2f} MB")
            print(f"  - Dashboard: http://{RAY_DASHBOARD_HOST}:{RAY_DASHBOARD_PORT}")
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize Ray cluster: {e}")
            self.shutdown()
            return False
    
    def shutdown(self) -> None:
        """Gracefully shutdown the Ray cluster."""
        if ray.is_initialized():
            ray.shutdown()
        self._cluster_initialized = False
        self._context = None
    
    def is_initialized(self) -> bool:
        """Check if Ray cluster is initialized."""
        return self._cluster_initialized and ray.is_initialized()
    
    def get_dashboard_url(self) -> str:
        """Get the Ray dashboard URL."""
        return f"http://{RAY_DASHBOARD_HOST}:{RAY_DASHBOARD_PORT}"
    
    def get_cluster_resources(self) -> Dict[str, float]:
        """Get current cluster resource availability."""
        if not self.is_initialized():
            return {}
        return dict(ray.cluster_resources())


def init_ray_cluster() -> RayClusterManager:
    """
    Convenience function to initialize Ray cluster.
    
    Returns:
        RayClusterManager instance
    """
    manager = RayClusterManager()
    if not manager.initialize():
        raise RuntimeError("Failed to initialize Ray cluster")
    return manager


def get_ray_context() -> Optional[ray.ClientContext]:
    """Get the current Ray context if initialized."""
    if ray.is_initialized():
        return ray.get_runtime_context()
    return None
