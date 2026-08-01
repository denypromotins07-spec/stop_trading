"""
Chapter 5: Extreme Python Optimization (Numba/Cython & GIL Bypass)
opt_mod.py - Module root managing JIT compilation cache and enforcing strict memory views
"""

import numpy as np
from typing import Dict, Optional, Tuple, List, Any, Callable
from dataclasses import dataclass, field
import threading
import weakref
from collections import OrderedDict

# Import local modules
from .numba_kernels import (
    NumbaKernelManager,
    zscore_kernel,
    ewma_kernel,
    covariance_matrix_kernel,
    rolling_mean_kernel,
    rolling_std_kernel,
    vwap_kernel,
    var_parametric_kernel,
    historical_var_kernel,
    expected_shortfall_kernel,
    matrix_multiply_kernel,
    cholesky_decomposition,
    create_kernel_manager
)
from .cffi_bridge import (
    CFFIBridge,
    CMemoryBlock,
    get_cffi_bridge,
    fast_matrix_multiply,
    fast_dot,
    fast_norm
)


@dataclass
class MemoryViewStats:
    """Statistics about memory view usage and copying prevention."""
    total_views_created: int = 0
    zero_copy_views: int = 0
    copied_views: int = 0
    bytes_saved: int = 0
    c_allocations: int = 0
    c_deallocations: int = 0


class MemoryViewManager:
    """
    Manages numpy memory views to prevent unnecessary array copying.
    Enforces contiguous memory layouts for optimal performance.
    """
    
    def __init__(self, max_cache_size: int = 100):
        self._cache: OrderedDict = OrderedDict()
        self._max_cache_size = max_cache_size
        self._lock = threading.Lock()
        self._stats = MemoryViewStats()
        
        # Weak references to track array lifetimes
        self._array_refs: Dict[int, weakref.ref] = {}
    
    def get_contiguous_view(
        self,
        arr: np.ndarray,
        dtype: Optional[np.dtype] = None,
        order: str = 'C'
    ) -> np.ndarray:
        """
        Get a contiguous memory view of an array.
        Avoids copying if array is already contiguous.
        
        Args:
            arr: Input array
            dtype: Target dtype (optional)
            order: Memory order ('C' or 'F')
        
        Returns:
            Contiguous view (may be copy if layout differs)
        """
        with self._lock:
            target_dtype = dtype if dtype is not None else arr.dtype
            
            # Check if already contiguous with correct dtype
            if arr.dtype == target_dtype and arr.flags['C_CONTIGUOUS']:
                self._stats.zero_copy_views += 1
                self._stats.total_views_created += 1
                return arr
            
            # Check if we can use ascontiguousarray without copy
            if arr.dtype == target_dtype:
                result = np.ascontiguousarray(arr, dtype=target_dtype)
                if np.shares_memory(arr, result):
                    self._stats.zero_copy_views += 1
                else:
                    self._stats.copied_views += 1
                
                self._stats.total_views_created += 1
                return result
            
            # Need to convert dtype - this requires a copy
            self._stats.copied_views += 1
            self._stats.total_views_created += 1
            return np.ascontiguousarray(arr, dtype=target_dtype)
    
    def get_memoryview(self, arr: np.ndarray) -> memoryview:
        """
        Get Python memoryview object for zero-copy access.
        
        Args:
            arr: NumPy array
        
        Returns:
            memoryview object
        """
        with self._lock:
            # Ensure contiguous first
            contiguous = self.get_contiguous_view(arr)
            
            # Create memoryview
            mv = memoryview(contiguous)
            self._stats.bytes_saved += mv.nbytes
            
            return mv
    
    def cache_array(
        self,
        key: str,
        arr: np.ndarray,
        ttl_seconds: float = 60.0
    ) -> None:
        """
        Cache an array for reuse.
        
        Args:
            key: Cache key
            arr: Array to cache
            ttl_seconds: Time to live in seconds
        """
        with self._lock:
            # Evict oldest if at capacity
            while len(self._cache) >= self._max_cache_size:
                self._cache.popitem(last=False)
            
            # Store contiguous version
            contiguous = self.get_contiguous_view(arr)
            self._cache[key] = contiguous
            
            # Track with weak reference
            arr_id = id(arr)
            self._array_refs[arr_id] = weakref.ref(
                arr, 
                lambda ref, k=key: self._on_array_collected(k)
            )
    
    def get_cached(self, key: str) -> Optional[np.ndarray]:
        """Get cached array by key."""
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                return self._cache[key]
        return None
    
    def _on_array_collected(self, key: str):
        """Callback when cached array is garbage collected."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
    
    def clear_cache(self):
        """Clear all cached arrays."""
        with self._lock:
            self._cache.clear()
            self._array_refs.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory view statistics."""
        with self._lock:
            return {
                'total_views': self._stats.total_views_created,
                'zero_copy': self._stats.zero_copy_views,
                'copied': self._stats.copied_views,
                'copy_ratio': self._stats.copied_views / max(1, self._stats.total_views_created),
                'bytes_saved': self._stats.bytes_saved,
                'cache_size': len(self._cache)
            }


class OptimizationModule:
    """
    Central module for managing JIT compilation and memory optimization.
    Coordinates Numba kernels and CFFI bridge for maximum performance.
    """
    
    def __init__(
        self,
        use_c_libs: bool = True,
        enable_cache: bool = True,
        max_cache_size: int = 100
    ):
        # Initialize kernel manager
        self.kernel_manager = create_kernel_manager()
        
        # Initialize CFFI bridge
        self.cffi_bridge = get_cffi_bridge(use_c_libs)
        
        # Initialize memory view manager
        self.memory_manager = MemoryViewManager(max_cache_size)
        
        # Compilation state
        self._compiled_functions: Dict[str, Callable] = {}
        self._compilation_times: Dict[str, float] = {}
        self._lock = threading.Lock()
        
        # GIL state tracking
        self._gil_released_count = 0
        self._gil_held_count = 0
        
        # Configuration
        self.enable_cache = enable_cache
        self.use_c_libs = use_c_libs
    
    def compute_zscore(
        self,
        data: np.ndarray,
        window: int
    ) -> np.ndarray:
        """Compute Z-scores using optimized kernel."""
        # Get contiguous view
        data_view = self.memory_manager.get_contiguous_view(data)
        
        # Use Numba kernel (releases GIL)
        self._gil_released_count += 1
        result = zscore_kernel(data_view, window)
        
        return result
    
    def compute_ewma(
        self,
        data: np.ndarray,
        alpha: float
    ) -> np.ndarray:
        """Compute EWMA using optimized kernel."""
        data_view = self.memory_manager.get_contiguous_view(data)
        
        self._gil_released_count += 1
        result = ewma_kernel(data_view, alpha)
        
        return result
    
    def compute_covariance_matrix(
        self,
        returns: np.ndarray
    ) -> np.ndarray:
        """Compute covariance matrix using parallel kernel."""
        returns_view = self.memory_manager.get_contiguous_view(returns)
        
        self._gil_released_count += 1
        result = covariance_matrix_kernel(returns_view)
        
        return result
    
    def matrix_multiply(
        self,
        A: np.ndarray,
        B: np.ndarray,
        use_cffi: bool = True
    ) -> np.ndarray:
        """
        High-performance matrix multiplication.
        
        Args:
            A: Left matrix
            B: Right matrix
            use_cffi: Whether to use CFFI bridge
        
        Returns:
            Result matrix
        """
        A_view = self.memory_manager.get_contiguous_view(A)
        B_view = self.memory_manager.get_contiguous_view(B)
        
        if use_cffi and self.use_c_libs:
            self._gil_released_count += 1
            return self.cffi_bridge.matrix_multiply(A_view, B_view)
        else:
            self._gil_released_count += 1
            return matrix_multiply_kernel(A_view, B_view)
    
    def dot_product(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> float:
        """High-performance dot product."""
        x_view = self.memory_manager.get_contiguous_view(x)
        y_view = self.memory_manager.get_contiguous_view(y)
        
        self._gil_released_count += 1
        return fast_dot(x_view, y_view)
    
    def vector_norm(
        self,
        x: np.ndarray
    ) -> float:
        """High-performance vector norm."""
        x_view = self.memory_manager.get_contiguous_view(x)
        
        self._gil_released_count += 1
        return fast_norm(x_view)
    
    def compute_risk_metrics(
        self,
        returns: np.ndarray,
        confidence: float = 0.99
    ) -> Dict[str, float]:
        """
        Compute comprehensive risk metrics using optimized kernels.
        
        Returns:
            Dictionary with VaR, CVaR, volatility
        """
        returns_view = self.memory_manager.get_contiguous_view(returns)
        
        self._gil_released_count += 3
        
        var_param = var_parametric_kernel(returns_view, confidence)
        var_hist = historical_var_kernel(returns_view, confidence)
        cvar = expected_shortfall_kernel(returns_view, confidence)
        
        return {
            'parametric_var': var_param,
            'historical_var': var_hist,
            'expected_shortfall': cvar,
            'volatility': np.std(returns_view)
        }
    
    def allocate_pinned_memory(
        self,
        shape: Tuple[int, ...],
        dtype: np.dtype = np.float64
    ) -> CMemoryBlock:
        """
        Allocate pinned (page-locked) memory for DMA transfers.
        
        Args:
            shape: Array shape
            dtype: Data type
        
        Returns:
            CMemoryBlock with numpy view
        """
        block = self.cffi_bridge.allocate_c_array(shape, dtype)
        self.memory_manager._stats.c_allocations += 1
        return block
    
    def register_compiled_function(
        self,
        name: str,
        func: Callable,
        compilation_time: float = 0.0
    ) -> None:
        """Register a compiled function for tracking."""
        with self._lock:
            self._compiled_functions[name] = func
            self._compilation_times[name] = compilation_time
    
    def get_compilation_stats(self) -> Dict[str, Any]:
        """Get JIT compilation statistics."""
        with self._lock:
            return {
                'compiled_functions': len(self._compiled_functions),
                'total_compilation_time': sum(self._compilation_times.values()),
                'gil_released_ops': self._gil_released_count,
                'gil_held_ops': self._gil_held_count,
                'cache_enabled': self.enable_cache,
                'c_libs_enabled': self.use_c_libs
            }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory management statistics."""
        mem_stats = self.memory_manager.get_stats()
        cffi_stats = self.cffi_bridge.get_memory_stats()
        
        return {
            **mem_stats,
            'cffi': cffi_stats
        }
    
    def clear_all_caches(self):
        """Clear all caches (JIT, memory, etc.)."""
        self.kernel_manager.clear_cache()
        self.memory_manager.clear_cache()
        self.cffi_bridge.cleanup()
        
        with self._lock:
            self._compiled_functions.clear()
            self._compilation_times.clear()
    
    def warmup_kernels(self):
        """Pre-compile all kernels to avoid runtime compilation latency."""
        # Small test arrays
        test_data = np.random.randn(100).astype(np.float64)
        test_returns = np.random.randn(100, 5).astype(np.float64)
        test_matrix = np.random.randn(50, 50).astype(np.float64)
        
        # Trigger compilation
        _ = zscore_kernel(test_data, 20)
        _ = ewma_kernel(test_data, 0.1)
        _ = covariance_matrix_kernel(test_returns)
        _ = matrix_multiply_kernel(test_matrix, test_matrix)
        _ = rolling_mean_kernel(test_data, 20)
        _ = rolling_std_kernel(test_data, 20)
        _ = vwap_kernel(test_data, np.abs(test_data))
        
        # Warm up CFFI bridge
        _ = self.cffi_bridge.matrix_multiply(test_matrix, test_matrix)
        _ = self.cffi_bridge.dot_product(test_data, test_data)


# Module singleton instance
_opt_module: Optional[OptimizationModule] = None


def get_optimization_module(
    use_c_libs: bool = True,
    enable_cache: bool = True
) -> OptimizationModule:
    """Get or create the global optimization module instance."""
    global _opt_module
    if _opt_module is None:
        _opt_module = OptimizationModule(use_c_libs, enable_cache)
    return _opt_module


def reset_optimization_module():
    """Reset the global optimization module (for testing)."""
    global _opt_module
    if _opt_module is not None:
        _opt_module.clear_all_caches()
    _opt_module = None


# Convenience functions
def quick_zscore(data: np.ndarray, window: int) -> np.ndarray:
    """Quick Z-score with automatic memory management."""
    opt = get_optimization_module()
    return opt.compute_zscore(data, window)


def quick_ewma(data: np.ndarray, alpha: float) -> np.ndarray:
    """Quick EWMA with automatic memory management."""
    opt = get_optimization_module()
    return opt.compute_ewma(data, alpha)


def quick_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Quick matrix multiply with automatic memory management."""
    opt = get_optimization_module()
    return opt.matrix_multiply(A, B)
