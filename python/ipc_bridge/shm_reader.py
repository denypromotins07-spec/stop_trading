"""
Zero-copy shared memory reader using mmap and numpy.frombuffer.
Ingests feature vectors directly from Rust without RAM duplication.
"""

import mmap
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Any
import sys
import os

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import SHM_SEGMENT_NAME, SHM_SEGMENT_SIZE, get_logger

logger = get_logger("shm_reader")


class SharedMemoryReader:
    """
    Zero-copy shared memory reader for ingesting feature vectors from Rust.
    Uses mmap and numpy.frombuffer to avoid any RAM duplication.
    """
    
    def __init__(
        self,
        segment_name: Optional[str] = None,
        segment_size: Optional[int] = None,
        read_only: bool = True,
    ):
        self.segment_name = segment_name or SHM_SEGMENT_NAME
        self.segment_size = segment_size or SHM_SEGMENT_SIZE
        self.read_only = read_only
        
        self._mmap: Optional[mmap.mmap] = None
        self._fd: Optional[int] = None
        self._is_open = False
        
        # Memoryview for zero-copy access
        self._memory_view: Optional[memoryview] = None
        
        # Numpy array views (zero-copy)
        self._feature_array: Optional[np.ndarray] = None
    
    def open(self) -> bool:
        """
        Open the shared memory segment for reading.
        
        Returns:
            True if successfully opened
        """
        if self._is_open:
            return True
        
        try:
            # Construct the path for POSIX shared memory
            shm_path = f"/dev/shm{self.segment_name}"
            
            # Check if file exists
            if not os.path.exists(shm_path):
                logger.warning(f"Shared memory segment {shm_path} does not exist yet")
                return False
            
            # Open the file descriptor
            flags = os.O_RDONLY if self.read_only else os.O_RDWR
            self._fd = os.open(shm_path, flags)
            
            # Create memory map
            prot = mmap.PROT_READ if self.read_only else (mmap.PROT_READ | mmap.PROT_WRITE)
            self._mmap = mmap.mmap(
                self._fd,
                self.segment_size,
                mmap.MAP_SHARED,
                prot,
            )
            
            # Create memoryview for zero-copy access
            self._memory_view = memoryview(self._mmap)
            
            self._is_open = True
            logger.info(f"Opened shared memory segment {self.segment_name} ({self.segment_size} bytes)")
            
            return True
            
        except FileNotFoundError:
            logger.error(f"Shared memory segment {self.segment_name} not found")
            return False
        except Exception as e:
            logger.error(f"Failed to open shared memory: {e}")
            self.close()
            return False
    
    def close(self) -> None:
        """Close the shared memory segment."""
        if self._mmap:
            try:
                self._mmap.close()
            except Exception:
                pass
            self._mmap = None
        
        if self._fd is not None:
            try:
                os.close(self._fd)
            except Exception:
                pass
            self._fd = None
        
        self._memory_view = None
        self._feature_array = None
        self._is_open = False
        logger.debug("Closed shared memory segment")
    
    def get_memoryview(self) -> Optional[memoryview]:
        """
        Get a memoryview of the shared memory for zero-copy access.
        
        Returns:
            memoryview object or None if not open
        """
        if not self._is_open:
            return None
        return self._memory_view
    
    def get_numpy_array(
        self,
        dtype: np.dtype = np.float64,
        shape: Optional[Tuple[int, ...]] = None,
        offset: int = 0,
    ) -> Optional[np.ndarray]:
        """
        Get a zero-copy numpy array view of the shared memory.
        
        Args:
            dtype: NumPy data type
            shape: Shape of the array (optional, calculates from size if not provided)
            offset: Byte offset into shared memory
        
        Returns:
            NumPy array view (zero-copy) or None if not open
        """
        if not self._is_open:
            return None
        
        try:
            # Calculate number of elements
            element_size = np.dtype(dtype).itemsize
            available_bytes = self.segment_size - offset
            num_elements = available_bytes // element_size
            
            if shape is None:
                shape = (num_elements,)
            else:
                # Verify shape matches available space
                expected_elements = np.prod(shape)
                if expected_elements * element_size > available_bytes:
                    logger.warning(f"Shape {shape} exceeds available shared memory")
                    return None
            
            # Create zero-copy numpy array from mmap buffer
            # This uses numpy.frombuffer which creates a view, not a copy
            arr = np.frombuffer(
                self._mmap,
                dtype=dtype,
                count=np.prod(shape),
                offset=offset,
            )
            
            # Reshape if needed
            if len(shape) > 1:
                arr = arr.reshape(shape)
            
            self._feature_array = arr
            return arr
            
        except Exception as e:
            logger.error(f"Failed to create numpy array view: {e}")
            return None
    
    def read_features(
        self,
        num_features: int,
        dtype: np.dtype = np.float64,
    ) -> Optional[np.ndarray]:
        """
        Read feature vector from shared memory.
        
        Args:
            num_features: Number of features to read
            dtype: Data type of features
        
        Returns:
            Feature array (zero-copy view) or None
        """
        return self.get_numpy_array(
            dtype=dtype,
            shape=(num_features,),
            offset=0,
        )
    
    def read_structured_data(
        self,
        dtype: np.dtype,
        count: int,
        offset: int = 0,
    ) -> Optional[np.ndarray]:
        """
        Read structured data from shared memory.
        
        Args:
            dtype: Structured dtype matching Rust layout
            count: Number of records
            offset: Byte offset
        
        Returns:
            Structured array (zero-copy view) or None
        """
        return self.get_numpy_array(
            dtype=dtype,
            shape=(count,),
            offset=offset,
        )
    
    def is_open(self) -> bool:
        """Check if shared memory is open."""
        return self._is_open
    
    def __enter__(self) -> "SharedMemoryReader":
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


# Global shared memory reader instance
_shm_reader_instance: Optional[SharedMemoryReader] = None


def get_shm_reader() -> SharedMemoryReader:
    """Get or create the global shared memory reader instance."""
    global _shm_reader_instance
    if _shm_reader_instance is None:
        _shm_reader_instance = SharedMemoryReader()
    return _shm_reader_instance


def read_features_zero_copy(num_features: int) -> Optional[np.ndarray]:
    """
    Convenience function to read features with zero-copy.
    
    Args:
        num_features: Number of features to read
    
    Returns:
        Feature array or None
    """
    reader = get_shm_reader()
    if not reader.is_open():
        if not reader.open():
            return None
    return reader.read_features(num_features)
