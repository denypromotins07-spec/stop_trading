"""
Chapter 5: Extreme Python Optimization (Numba/Cython & GIL Bypass)
cffi_bridge.py - CFFI bridge to call ultra-fast pre-compiled C math libraries directly from Python
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
import ctypes
from dataclasses import dataclass

# Try to import cffi, provide graceful fallback
try:
    from cffi import FFI
    CFFI_AVAILABLE = True
except ImportError:
    CFFI_AVAILABLE = False


# ============================================================================
# C Library Definitions
# ============================================================================

C_MATH_DEFINITIONS = """
// Basic BLAS-like operations
void c_daxpy(int n, double alpha, const double *x, int incx, double *y, int incy);
double c_ddot(int n, const double *x, int incx, const double *y, int incy);
void c_dscal(int n, double alpha, double *x, int incx);
double c_dnrm2(int n, const double *x, int incx);

// Matrix operations
void c_dgemm(char transa, char transb, int m, int n, int k, 
             double alpha, const double *a, int lda,
             const double *b, int ldb, double beta, double *c, int ldc);

// Linear algebra
int c_dposv(char uplo, int n, int nrhs, double *a, int lda, 
            double *b, int ldb, int *info);

// Memory management
void* c_malloc(size_t size);
void c_free(void *ptr);
"""

# Pure Python/CFFI implementations for when external C libs unavailable
class FallbackCLibrary:
    """
    Pure Python fallback implementations of C math functions.
    Uses Numba JIT for performance.
    """
    
    @staticmethod
    def daxpy(n: int, alpha: float, x: np.ndarray, y: np.ndarray) -> None:
        """y = alpha * x + y (BLAS DAXPY)"""
        try:
            from numba import njit
            
            @njit(cache=True, nogil=True)
            def _daxpy_impl(n, alpha, x, y):
                for i in range(n):
                    y[i] = alpha * x[i] + y[i]
            
            _daxpy_impl(n, alpha, x, y)
        except ImportError:
            for i in range(min(n, len(x), len(y))):
                y[i] = alpha * x[i] + y[i]
    
    @staticmethod
    def ddot(n: int, x: np.ndarray, y: np.ndarray) -> float:
        """Dot product (BLAS DDOT)"""
        try:
            from numba import njit
            
            @njit(cache=True, nogil=True)
            def _ddot_impl(n, x, y):
                result = 0.0
                for i in range(n):
                    result += x[i] * y[i]
                return result
            
            return _ddot_impl(n, x, y)
        except ImportError:
            return float(np.dot(x[:n], y[:n]))
    
    @staticmethod
    def dscal(n: int, alpha: float, x: np.ndarray) -> None:
        """x = alpha * x (BLAS DSCAL)"""
        try:
            from numba import njit
            
            @njit(cache=True, nogil=True)
            def _dscal_impl(n, alpha, x):
                for i in range(n):
                    x[i] = alpha * x[i]
            
            _dscal_impl(n, alpha, x)
        except ImportError:
            x[:n] *= alpha
    
    @staticmethod
    def dnrm2(n: int, x: np.ndarray) -> float:
        """Euclidean norm (BLAS DNRM2)"""
        try:
            from numba import njit
            
            @njit(cache=True, nogil=True)
            def _dnrm2_impl(n, x):
                sum_sq = 0.0
                for i in range(n):
                    sum_sq += x[i] * x[i]
                return np.sqrt(sum_sq)
            
            return _dnrm2_impl(n, x)
        except ImportError:
            return float(np.linalg.norm(x[:n]))
    
    @staticmethod
    def dgemm(
        A: np.ndarray,
        B: np.ndarray,
        alpha: float = 1.0,
        beta: float = 0.0,
        C: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Matrix multiplication (BLAS DGEMM): C = alpha * A * B + beta * C"""
        if C is None:
            C = np.zeros((A.shape[0], B.shape[1]), dtype=np.float64)
        
        try:
            from numba import njit, prange
            
            @njit(cache=True, nogil=True, parallel=True)
            def _dgemm_impl(A, B, alpha, beta, C):
                m, k = A.shape
                k2, n = B.shape
                
                # Apply beta first
                if beta != 0.0:
                    for i in range(m):
                        for j in range(n):
                            C[i, j] = beta * C[i, j]
                elif beta == 0.0:
                    for i in range(m):
                        for j in range(n):
                            C[i, j] = 0.0
                
                # Matrix multiply
                for i in prange(m):
                    for j in range(n):
                        s = 0.0
                        for l in range(k):
                            s += A[i, l] * B[l, j]
                        C[i, j] += alpha * s
                
                return C
            
            return _dgemm_impl(A, B, alpha, beta, C)
        except ImportError:
            return alpha * np.dot(A, B) + beta * C


@dataclass
class CMemoryBlock:
    """Represents a C-allocated memory block with Python view."""
    ptr: int  # Memory address
    size: int  # Size in bytes
    dtype: np.dtype
    shape: Tuple[int, ...]
    _owner: Any  # Reference to owner for GC
    
    def to_numpy(self) -> np.ndarray:
        """Create numpy array view of C memory without copying."""
        if not CFFI_AVAILABLE:
            raise RuntimeError("CFFI not available")
        
        ffi = FFI()
        array = ffi.from_buffer(self.dtype, ffi.cast(f"{self.dtype.char}*", self.ptr), self.size)
        return array.reshape(self.shape)
    
    def __del__(self):
        """Free C memory when Python object is garbage collected."""
        if CFFI_AVAILABLE and self.ptr != 0:
            ffi = FFI()
            ffi.cdef("void free(void *ptr);")
            ffi.CLibrary().free(ffi.cast("void *", self.ptr))


class CFFIBridge:
    """
    Bridge between Python/numpy and C math libraries.
    Handles memory ownership to prevent premature GC of C buffers.
    """
    
    def __init__(self, use_c_libs: bool = True):
        self.use_c_libs = use_c_libs and CFFI_AVAILABLE
        self._ffi = FFI() if CFFI_AVAILABLE else None
        self._clib = None
        self._allocated_blocks: Dict[int, CMemoryBlock] = {}
        self._fallback = FallbackCLibrary()
        
        if self.use_c_libs:
            self._setup_c_library()
    
    def _setup_c_library(self):
        """Set up C library interface."""
        if not CFFI_AVAILABLE:
            return
        
        self._ffi.cdef(C_MATH_DEFINITIONS)
        
        try:
            # Try to load system BLAS/LAPACK
            self._clib = self._ffi.dlopen(None)  # Standard C library
        except Exception:
            self.use_c_libs = False
    
    def allocate_c_array(
        self,
        shape: Tuple[int, ...],
        dtype: np.dtype = np.float64
    ) -> CMemoryBlock:
        """
        Allocate memory on C heap for zero-copy operations.
        
        Args:
            shape: Array shape
            dtype: NumPy dtype
        
        Returns:
            CMemoryBlock with numpy view
        """
        if not CFFI_AVAILABLE:
            # Fallback: use numpy array
            arr = np.empty(shape, dtype=dtype)
            return CMemoryBlock(
                ptr=arr.ctypes.data,
                size=arr.nbytes,
                dtype=dtype,
                shape=shape,
                _owner=arr
            )
        
        n_elements = int(np.prod(shape))
        size_bytes = n_elements * dtype.itemsize
        
        ptr = self._ffi.new(f"{dtype.__name__}[]", n_elements)
        addr = int(self._ffi.cast("uintptr_t", ptr))
        
        block = CMemoryBlock(
            ptr=addr,
            size=size_bytes,
            dtype=dtype,
            shape=shape,
            _owner=ptr  # Keep reference to prevent GC
        )
        
        self._allocated_blocks[addr] = block
        
        return block
    
    def matrix_multiply(
        self,
        A: np.ndarray,
        B: np.ndarray,
        alpha: float = 1.0,
        beta: float = 0.0,
        C: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        High-performance matrix multiplication via C or fallback.
        
        Args:
            A: Left matrix
            B: Right matrix
            alpha: Scaling factor for A*B
            beta: Scaling factor for existing C
            C: Output matrix (optional, allocated if None)
        
        Returns:
            Result matrix
        """
        # Ensure contiguous arrays
        A = np.ascontiguousarray(A, dtype=np.float64)
        B = np.ascontiguousarray(B, dtype=np.float64)
        
        if C is not None:
            C = np.ascontiguousarray(C, dtype=np.float64)
        
        if self.use_c_libs and self._clib is not None:
            # Use C implementation via CFFI
            return self._cblas_gemm(A, B, alpha, beta, C)
        else:
            # Use Numba fallback
            return self._fallback.dgemm(A, B, alpha, beta, C)
    
    def _cblas_gemm(
        self,
        A: np.ndarray,
        B: np.ndarray,
        alpha: float,
        beta: float,
        C: Optional[np.ndarray]
    ) -> np.ndarray:
        """C implementation of DGEMM via CFFI."""
        m, k = A.shape
        k2, n = B.shape
        
        if C is None:
            C = np.zeros((m, n), dtype=np.float64)
        
        # Get pointers
        A_ptr = self._ffi.from_buffer("double *", A)
        B_ptr = self._ffi.from_buffer("double *", B)
        C_ptr = self._ffi.from_buffer("double *", C)
        
        # Call DGEMM
        self._clib.c_dgemm(
            b'N', b'N',
            m, n, k,
            alpha,
            A_ptr, m,
            B_ptr, k2,
            beta,
            C_ptr, m
        )
        
        return C
    
    def dot_product(self, x: np.ndarray, y: np.ndarray) -> float:
        """High-performance dot product."""
        x = np.ascontiguousarray(x, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        
        n = min(len(x), len(y))
        
        if self.use_c_libs and self._clib is not None:
            x_ptr = self._ffi.from_buffer("double *", x)
            y_ptr = self._ffi.from_buffer("double *", y)
            return self._clib.c_ddot(n, x_ptr, 1, y_ptr, 1)
        else:
            return self._fallback.ddot(n, x, y)
    
    def axpy(self, alpha: float, x: np.ndarray, y: np.ndarray) -> None:
        """y = alpha * x + y (in-place)."""
        x = np.ascontiguousarray(x, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        
        n = min(len(x), len(y))
        
        if self.use_c_libs and self._clib is not None:
            x_ptr = self._ffi.from_buffer("double *", x)
            y_ptr = self._ffi.from_buffer("double *", y)
            self._clib.c_daxpy(n, alpha, x_ptr, 1, y_ptr, 1)
        else:
            self._fallback.daxpy(n, alpha, x, y)
    
    def scale(self, alpha: float, x: np.ndarray) -> None:
        """x = alpha * x (in-place)."""
        x = np.ascontiguousarray(x, dtype=np.float64)
        n = len(x)
        
        if self.use_c_libs and self._clib is not None:
            x_ptr = self._ffi.from_buffer("double *", x)
            self._clib.c_dscal(n, alpha, x_ptr, 1)
        else:
            self._fallback.dscal(n, alpha, x)
    
    def norm(self, x: np.ndarray) -> float:
        """Euclidean norm."""
        x = np.ascontiguousarray(x, dtype=np.float64)
        n = len(x)
        
        if self.use_c_libs and self._clib is not None:
            x_ptr = self._ffi.from_buffer("double *", x)
            return self._clib.c_dnrm2(n, x_ptr, 1)
        else:
            return self._fallback.dnrm2(n, x)
    
    def solve_positive_definite(
        self,
        A: np.ndarray,
        B: np.ndarray
    ) -> np.ndarray:
        """
        Solve AX = B where A is positive definite.
        Uses Cholesky decomposition internally.
        """
        # Fallback to numpy/scipy since LAPACK is complex to link
        A = np.ascontiguousarray(A, dtype=np.float64)
        B = np.ascontiguousarray(B, dtype=np.float64)
        
        try:
            # Use scipy if available
            from scipy.linalg import cho_solve, cholesky
            L = cholesky(A, lower=True)
            return cho_solve((L, True), B)
        except ImportError:
            # Pure numpy fallback
            return np.linalg.solve(A, B)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about C memory allocations."""
        return {
            'allocated_blocks': len(self._allocated_blocks),
            'total_bytes': sum(b.size for b in self._allocated_blocks.values()),
            'using_c_libs': self.use_c_libs,
            'cffi_available': CFFI_AVAILABLE
        }
    
    def cleanup(self):
        """Clean up all C allocations."""
        self._allocated_blocks.clear()


# Module singleton instance
_bridge: Optional[CFFIBridge] = None


def get_cffi_bridge(use_c_libs: bool = True) -> CFFIBridge:
    """Get or create the global CFFI bridge instance."""
    global _bridge
    if _bridge is None:
        _bridge = CFFIBridge(use_c_libs)
    return _bridge


def reset_cffi_bridge():
    """Reset the global bridge (for testing)."""
    global _bridge
    if _bridge is not None:
        _bridge.cleanup()
    _bridge = None


# Convenience functions
def fast_matrix_multiply(
    A: np.ndarray,
    B: np.ndarray
) -> np.ndarray:
    """Quick matrix multiplication using best available backend."""
    bridge = get_cffi_bridge()
    return bridge.matrix_multiply(A, B)


def fast_dot(x: np.ndarray, y: np.ndarray) -> float:
    """Quick dot product using best available backend."""
    bridge = get_cffi_bridge()
    return bridge.dot_product(x, y)


def fast_norm(x: np.ndarray) -> float:
    """Quick vector norm using best available backend."""
    bridge = get_cffi_bridge()
    return bridge.norm(x)
