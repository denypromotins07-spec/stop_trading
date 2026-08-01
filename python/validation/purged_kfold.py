"""
Purged K-Fold Cross-Validation for Time-Series ML.
Implements embargo periods to prevent data leakage in financial time-series.
Based on Lopez de Prado's methodology for avoiding look-ahead bias.
"""

import logging
from typing import List, Tuple, Generator, Optional, Dict, Any
import numpy as np
from sklearn.model_selection import BaseCrossValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PurgedKFold(BaseCrossValidator):
    """
    Purged K-Fold Cross-Validation with embargo periods.
    
    Prevents data leakage by:
    1. Removing training samples that overlap with test samples in time
    2. Adding an embargo period after each test fold
    
    Essential for time-series ML where observations are not i.i.d.
    """
    
    def __init__(self, 
                 n_splits: int = 5,
                 embargo_pct: float = 0.01,
                 samples_info: Optional[Dict[int, Tuple[float, float]]] = None,
                 shuffle: bool = False,
                 random_state: Optional[int] = None):
        """
        Initialize Purged K-Fold.
        
        Args:
            n_splits: Number of folds
            embargo_pct: Embargo period as percentage of total samples
            samples_info: Optional dict mapping sample_idx -> (start_time, end_time)
                         If None, assumes sequential ordering with unit duration
            shuffle: Whether to shuffle data before splitting (not recommended for time-series)
            random_state: Random seed for shuffling
        """
        self.n_splits = n_splits
        self.embargo_pct = max(0.0, min(1.0, embargo_pct))
        self.samples_info = samples_info
        self.shuffle = shuffle
        self.random_state = random_state
        
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
    
    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None,
              groups: Optional[np.ndarray] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices with purging and embargo.
        
        Args:
            X: Feature matrix (only shape is used)
            y: Target vector (optional)
            groups: Group labels (optional, not used)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Shuffle if requested (usually not for time-series)
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            indices = rng.permutation(indices)
        
        # Calculate fold boundaries
        fold_size = n_samples // self.n_splits
        remainder = n_samples % self.n_splits
        
        # Calculate embargo size in samples
        embargo_size = int(np.ceil(n_samples * self.embargo_pct))
        
        # Build time intervals if not provided
        if self.samples_info is None:
            # Assume sequential with unit duration
            self.samples_info = {i: (float(i), float(i + 1)) for i in range(n_samples)}
        
        # Generate folds
        start_idx = 0
        for fold in range(self.n_splits):
            # Add remainder to first folds
            current_fold_size = fold_size + (1 if fold < remainder else 0)
            
            # Test indices for this fold
            test_start = start_idx
            test_end = start_idx + current_fold_size
            test_indices = indices[test_start:test_end]
            
            # Get time range of test set
            test_times = [self.samples_info[i] for i in test_indices if i in self.samples_info]
            if test_times:
                test_min_time = min(t[0] for t in test_times)
                test_max_time = max(t[1] for t in test_times)
            else:
                test_min_time = float(test_start)
                test_max_time = float(test_end)
            
            # Determine purge and embargo regions
            train_indices = []
            for i in indices:
                if i in test_indices:
                    continue  # Skip test samples
                
                sample_time = self.samples_info.get(i, (float(i), float(i + 1)))
                sample_start, sample_end = sample_time
                
                # Purge: Remove if sample overlaps with test period
                if sample_end > test_min_time and sample_start < test_max_time:
                    continue  # Overlaps, purge this sample
                
                # Embargo: Remove if sample is within embargo period after test
                if sample_start > test_max_time:
                    # Check if within embargo window
                    embargo_threshold = test_max_time + (test_max_time - test_min_time) * self.embargo_pct / 0.1
                    if sample_start < embargo_threshold:
                        continue  # In embargo period
                
                train_indices.append(i)
            
            train_indices = np.array(train_indices, dtype=int)
            test_indices = np.array(test_indices, dtype=int)
            
            if len(train_indices) == 0:
                logger.warning(f"Fold {fold}: No training samples after purging!")
                continue
            
            yield train_indices, test_indices
            
            start_idx = test_end
    
    def get_n_splits(self, X: np.ndarray = None, y: np.ndarray = None,
                     groups: np.ndarray = None) -> int:
        """Return number of splits."""
        return self.n_splits
    
    def get_embargo_size(self, n_samples: int) -> int:
        """Calculate embargo size in samples."""
        return int(np.ceil(n_samples * self.embargo_pct))


def calculate_embargo_period(holding_times: np.ndarray, 
                             confidence_level: float = 0.95) -> float:
    """
    Calculate optimal embargo period based on average trade holding time.
    
    Args:
        holding_times: Array of trade holding times (in seconds or bars)
        confidence_level: Confidence level for percentile calculation
        
    Returns:
        Embargo period as fraction of average holding time
    """
    if len(holding_times) == 0:
        return 0.01  # Default 1%
    
    # Use percentile of holding times to determine embargo
    percentile_value = np.percentile(holding_times, confidence_level * 100)
    mean_holding = np.mean(holding_times)
    
    # Embargo should cover most holding periods to prevent leakage
    embargo_ratio = percentile_value / (mean_holding if mean_holding > 0 else 1)
    
    # Cap at reasonable bounds
    embargo_ratio = min(max(embargo_ratio, 0.001), 0.1)
    
    logger.info(f"Calculated embargo ratio: {embargo_ratio:.4f} "
               f"(p{confidence_level*100:.0f} holding time: {percentile_value:.2f})")
    
    return embargo_ratio


def create_purged_kfold_from_trades(trade_log: List[Dict[str, Any]],
                                    n_splits: int = 5,
                                    confidence_level: float = 0.95) -> PurgedKFold:
    """
    Create PurgedKFold from trade log with automatic embargo calculation.
    
    Args:
        trade_log: List of trade dictionaries with 'entry_time' and 'exit_time'
        n_splits: Number of CV folds
        confidence_level: Confidence level for embargo calculation
        
    Returns:
        Configured PurgedKFold instance
    """
    if not trade_log:
        return PurgedKFold(n_splits=n_splits)
    
    # Extract holding times
    holding_times = []
    samples_info = {}
    
    for i, trade in enumerate(trade_log):
        entry = trade.get('entry_time', i)
        exit_time = trade.get('exit_time', i + 1)
        holding_time = exit_time - entry
        
        holding_times.append(holding_time)
        samples_info[i] = (float(entry), float(exit_time))
    
    holding_times = np.array(holding_times)
    
    # Calculate embargo
    embargo_pct = calculate_embargo_period(holding_times, confidence_level)
    
    logger.info(f"Creating PurgedKFold with {n_splits} splits, "
               f"embargo={embargo_pct:.4f} ({len(trade_log)} trades)")
    
    return PurgedKFold(
        n_splits=n_splits,
        embargo_pct=embargo_pct,
        samples_info=samples_info
    )


if __name__ == "__main__":
    # Test PurgedKFold
    np.random.seed(42)
    
    # Simulate time-series data
    n_samples = 1000
    X = np.random.randn(n_samples, 10)
    y = np.random.randint(0, 2, n_samples)
    
    # Create samples info with varying durations
    samples_info = {}
    for i in range(n_samples):
        duration = np.random.exponential(2)  # Random holding times
        samples_info[i] = (float(i), float(i + duration))
    
    # Create purged k-fold
    pkf = PurgedKFold(n_splits=5, embargo_pct=0.02, samples_info=samples_info)
    
    print(f"Purged K-Fold: {pkf.n_splits} splits, embargo={pkf.embargo_pct:.2%}")
    print(f"Total samples: {n_samples}")
    print()
    
    # Iterate through folds
    for fold, (train_idx, test_idx) in enumerate(pkf.split(X)):
        train_size = len(train_idx)
        test_size = len(test_idx)
        gap_size = n_samples - train_size - test_size
        
        print(f"Fold {fold + 1}:")
        print(f"  Train: {train_size} samples ({train_size/n_samples:.1%})")
        print(f"  Test:  {test_size} samples ({test_size/n_samples:.1%})")
        print(f"  Gap:   {gap_size} samples purged/embargoed ({gap_size/n_samples:.1%})")
        print()
    
    # Test with trade log
    print("\n--- Trade-based PurgedKFold ---\n")
    
    trade_log = []
    current_time = 0
    for i in range(500):
        entry = current_time
        holding = np.random.exponential(5)
        exit_time = entry + holding
        trade_log.append({
            'entry_time': entry,
            'exit_time': exit_time,
            'trade_id': i
        })
        current_time = exit_time + np.random.exponential(1)
    
    pkf_trades = create_purged_kfold_from_trades(trade_log, n_splits=5)
    print(f"Created from {len(trade_log)} trades")
    print(f"Embargo: {pkf_trades.embargo_pct:.4f}")
