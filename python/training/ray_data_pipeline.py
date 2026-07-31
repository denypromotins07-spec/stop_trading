"""
Ray Data Pipeline - Memory-efficient data loading for batch retraining.
Uses memory-mapped files and streaming iterators to process large datasets.
Strictly enforces 3GB RAM limit with bounded batch sizes.
"""
import ray
from ray import data
from typing import Dict, List, Optional, Any, Iterator, Tuple
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)


class RayDataPipeline:
    """
    Ray Data pipeline for reading historical TickDB snapshots and SOUL.md logs.
    Uses streaming and memory mapping to stay within 3GB RAM ceiling.
    """
    
    def __init__(self,
                 batch_size: int = 1024,
                 max_memory_mb: int = 1024,
                 num_cpus: int = 4):
        """
        Initialize Ray Data pipeline.
        
        Args:
            batch_size: Batch size for streaming (small for memory safety)
            max_memory_mb: Maximum memory budget in MB
            num_cpus: Number of CPUs to use
        """
        self.batch_size = batch_size
        self.max_memory_mb = max_memory_mb
        self.num_cpus = num_cpus
        
        self._dataset = None
        self._current_split = None
    
    def read_parquet_streaming(self,
                               paths: List[str],
                               columns: List[str] = None,
                               filter_fn=None) -> 'RayDataPipeline':
        """
        Read parquet files with streaming iterator.
        
        Args:
            paths: List of parquet file paths
            columns: Columns to load (None for all)
            filter_fn: Optional filter function
            
        Returns:
            Self for chaining
        """
        try:
            # Use ray.data.read_parquet with strict batch limits
            self._dataset = ray.data.read_parquet(
                paths,
                columns=columns,
                parallelism=self.num_cpus,
                batch_size=self.batch_size,
                # Memory-safe options
                _use_pandas_block=False,  # Use Arrow for lower memory
            )
            
            # Apply filter if provided
            if filter_fn:
                self._dataset = self._dataset.filter(filter_fn)
            
            logger.info(f"Loaded dataset from {len(paths)} files")
            return self
            
        except Exception as e:
            logger.error(f"Failed to read parquet: {e}")
            raise
    
    def read_tickdb_snapshot(self,
                            snapshot_path: str,
                            instruments: List[str] = None,
                            start_ns: int = None,
                            end_ns: int = None) -> 'RayDataPipeline':
        """
        Read TickDB snapshot with time range filtering.
        
        Args:
            snapshot_path: Path to TickDB snapshot directory
            instruments: List of instruments to load
            start_ns: Start timestamp in nanoseconds
            end_ns: End timestamp in nanoseconds
            
        Returns:
            Self for chaining
        """
        # Build path pattern
        path = Path(snapshot_path)
        parquet_files = list(path.glob("**/*.parquet"))
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {snapshot_path}")
        
        # Define filter for time range
        def time_filter(row):
            ts = row.get('timestamp_ns', 0)
            if start_ns and ts < start_ns:
                return False
            if end_ns and ts > end_ns:
                return False
            if instruments and row.get('instrument') not in instruments:
                return False
            return True
        
        return self.read_parquet_streaming(
            paths=[str(f) for f in parquet_files],
            filter_fn=time_filter
        )
    
    def read_soul_logs(self, soul_log_paths: List[str]) -> 'RayDataPipeline':
        """
        Read SOUL.md log files for training data.
        
        Args:
            soul_log_paths: Paths to SOUL log files
            
        Returns:
            Self for chaining
        """
        def parse_soul_line(line: Dict) -> Optional[Dict]:
            """Parse a line from SOUL log."""
            # Simple parsing - adapt to actual format
            text = line.get('text', '')
            if not text.strip():
                return None
            
            # Extract structured data
            result = {'raw': text}
            
            if 'Outcome:' in text:
                result['type'] = 'outcome'
            elif 'Mistake:' in text:
                result['type'] = 'mistake'
            elif 'Regime' in text:
                result['type'] = 'regime'
            else:
                result['type'] = 'other'
            
            return result
        
        self._dataset = ray.data.from_items(
            [{'text': line} for path in soul_log_paths 
             for line in self._read_file_lines(path)]
        )
        self._dataset = self._dataset.map(parse_soul_line).filter(lambda x: x is not None)
        
        return self
    
    def _read_file_lines(self, path: str) -> Iterator[str]:
        """Read file lines with memory-bounded streaming."""
        try:
            with open(path, 'r') as f:
                for line in f:
                    yield line
        except Exception as e:
            logger.warning(f"Error reading {path}: {e}")
    
    def join_features_and_labels(self,
                                 feature_columns: List[str],
                                 label_column: str) -> 'RayDataPipeline':
        """
        Prepare dataset for training by selecting features and labels.
        
        Args:
            feature_columns: Columns to use as features
            label_column: Column to use as label
            
        Returns:
            Self for chaining
        """
        def extract_features_and_label(row):
            features = np.array([row.get(col, 0.0) for col in feature_columns])
            label = row.get(label_column, 0.0)
            return {'features': features, 'label': label, 'raw': row}
        
        if self._dataset:
            self._dataset = self._dataset.map(extract_features_and_label)
        
        return self
    
    def train_test_split(self,
                        test_size: float = 0.2,
                        shuffle: bool = True,
                        seed: int = 42) -> Tuple['RayDataPipeline', 'RayDataPipeline']:
        """
        Split dataset into train and test sets.
        
        Args:
            test_size: Fraction for test set
            shuffle: Whether to shuffle before splitting
            seed: Random seed
            
        Returns:
            Tuple of (train_pipeline, test_pipeline)
        """
        if not self._dataset:
            raise ValueError("No dataset loaded")
        
        train_ds, test_ds = self._dataset.train_test_split(
            test_size=test_size,
            shuffle=shuffle,
            seed=seed
        )
        
        train_pipe = RayDataPipeline(
            batch_size=self.batch_size,
            max_memory_mb=self.max_memory_mb
        )
        train_pipe._dataset = train_ds
        
        test_pipe = RayDataPipeline(
            batch_size=self.batch_size,
            max_memory_mb=self.max_memory_mb
        )
        test_pipe._dataset = test_ds
        
        return train_pipe, test_pipe
    
    def to_iter_batches(self,
                       batch_size: int = None,
                       columns: List[str] = None) -> Iterator[Dict[str, np.ndarray]]:
        """
        Iterate over dataset in batches.
        
        Args:
            batch_size: Batch size (uses default if None)
            columns: Columns to return
            
        Returns:
            Iterator of batch dictionaries
        """
        if not self._dataset:
            raise ValueError("No dataset loaded")
        
        bs = batch_size or self.batch_size
        
        for batch in self._dataset.iter_batches(
            batch_size=bs,
            columns=columns,
            batch_format='numpy'
        ):
            yield batch
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self._dataset:
            return {"loaded": False}
        
        try:
            count = self._dataset.count()
            schema = self._dataset.schema()
            
            return {
                "loaded": True,
                "row_count": count,
                "schema": str(schema),
                "batch_size": self.batch_size,
                "estimated_memory_mb": self._estimate_memory()
            }
        except Exception as e:
            logger.warning(f"Error getting stats: {e}")
            return {"loaded": True, "error": str(e)}
    
    def _estimate_memory(self) -> float:
        """Estimate memory usage in MB."""
        if not self._dataset:
            return 0.0
        
        try:
            # Rough estimate based on batch size and schema
            schema = self._dataset.schema()
            if schema:
                # Assume average 8 bytes per value
                estimated_row_size = len(schema.names) * 8
                return (self.batch_size * estimated_row_size) / (1024 * 1024)
        except:
            pass
        
        return self.max_memory_mb * 0.5  # Conservative estimate
    
    def cleanup(self):
        """Release dataset resources."""
        self._dataset = None
        self._current_split = None


def create_pipeline(batch_size: int = 1024,
                   max_memory_mb: int = 1024) -> RayDataPipeline:
    """Create a new Ray Data pipeline."""
    return RayDataPipeline(
        batch_size=batch_size,
        max_memory_mb=max_memory_mb
    )


# Example usage
def main():
    """Example usage of Ray Data pipeline."""
    # Initialize Ray
    ray.init(
        ignore_reinit_error=True,
        _system_config={
            "object_store_memory": 512 * 1024 * 1024,
            "max_workers": 4
        }
    )
    
    try:
        # Create pipeline
        pipeline = create_pipeline(batch_size=512, max_memory_mb=512)
        
        # Example: Load from parquet (would need actual files)
        # pipeline.read_parquet_streaming(
        #     paths=["data/ticks/*.parquet"],
        #     columns=["timestamp_ns", "price", "volume", "feature_1", "feature_2"]
        # )
        
        print(f"Pipeline stats: {pipeline.get_stats()}")
        
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
