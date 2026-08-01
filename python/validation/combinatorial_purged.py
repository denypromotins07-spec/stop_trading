"""
Combinatorial Purged Cross-Validation (CPCV).
Generates multiple backtest paths for robust Sharpe estimation.
Based on Lopez de Prado's CPCV methodology for avoiding overfitting in financial ML.
"""

import logging
from typing import List, Tuple, Generator, Optional, Dict, Any, Iterator
from itertools import combinations
from collections import defaultdict
import numpy as np
from scipy.stats import sem
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CombinatorialPurgedKFold:
    """
    Combinatorial Purged K-Fold Cross-Validation.
    
    Generates C(n_splits, n_test_folds) different train/test combinations,
    creating multiple backtest paths for robust performance estimation.
    
    This allows estimating the variance of strategy performance across
    different market regimes without look-ahead bias.
    """
    
    def __init__(self,
                 n_splits: int = 6,
                 n_test_folds: int = 2,
                 embargo_pct: float = 0.01,
                 samples_info: Optional[Dict[int, Tuple[float, float]]] = None,
                 random_state: Optional[int] = None):
        """
        Initialize CPCV.
        
        Args:
            n_splits: Total number of folds to divide data into
            n_test_folds: Number of folds to use as test set in each split
                         Must be < n_splits
            embargo_pct: Embargo period as percentage of total samples
            samples_info: Dict mapping sample_idx -> (start_time, end_time)
            random_state: Random seed for reproducibility
        """
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if n_test_folds < 1 or n_test_folds >= n_splits:
            raise ValueError("n_test_folds must be in [1, n_splits)")
        
        self.n_splits = n_splits
        self.n_test_folds = n_test_folds
        self.embargo_pct = max(0.0, min(1.0, embargo_pct))
        self.samples_info = samples_info
        self.random_state = random_state
        
        # Calculate number of combinations
        self.n_combinations = int(len(list(combinations(range(n_splits), n_test_folds))))
        
        logger.info(f"CPCV initialized: {self.n_splits} splits, "
                   f"{self.n_test_folds} test folds, {self.n_combinations} combinations")
    
    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Generator[Tuple[np.ndarray, np.ndarray, int], None, None]:
        """
        Generate train/test indices for all combinations.
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            
        Yields:
            Tuple of (train_indices, test_indices, combination_id)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Calculate fold sizes
        fold_size = n_samples // self.n_splits
        remainder = n_samples % self.n_splits
        
        # Build fold boundaries
        fold_boundaries = []
        start = 0
        for i in range(self.n_splits):
            size = fold_size + (1 if i < remainder else 0)
            fold_boundaries.append((start, start + size))
            start += size
        
        # Build samples_info if not provided
        if self.samples_info is None:
            self.samples_info = {}
            for fold_idx, (f_start, f_end) in enumerate(fold_boundaries):
                for i in range(f_start, f_end):
                    self.samples_info[i] = (float(i), float(i + 1))
        
        # Generate all combinations of test folds
        test_fold_combos = list(combinations(range(self.n_splits), self.n_test_folds))
        
        for combo_id, test_folds in enumerate(test_fold_combos):
            # Test indices: union of selected test folds
            test_indices = []
            test_time_ranges = []
            for fold_idx in test_folds:
                f_start, f_end = fold_boundaries[fold_idx]
                fold_indices = indices[f_start:f_end]
                test_indices.extend(fold_indices)
                
                # Get time range for this fold
                fold_times = [self.samples_info.get(i, (float(i), float(i+1))) for i in fold_indices]
                if fold_times:
                    test_time_ranges.append((min(t[0] for t in fold_times), max(t[1] for t in fold_times)))
            
            test_indices = np.array(test_indices, dtype=int)
            
            # Combined test time range
            if test_time_ranges:
                test_min_time = min(tr[0] for tr in test_time_ranges)
                test_max_time = max(tr[1] for tr in test_time_ranges)
            else:
                test_min_time = float(np.min(test_indices))
                test_max_time = float(np.max(test_indices) + 1)
            
            # Train indices: all other folds with purging and embargo
            train_indices = []
            for i in indices:
                if i in test_indices:
                    continue
                
                sample_time = self.samples_info.get(i, (float(i), float(i + 1)))
                sample_start, sample_end = sample_time
                
                # Purge: Remove overlapping samples
                if sample_end > test_min_time and sample_start < test_max_time:
                    continue
                
                # Embargo: Remove samples shortly after test period
                if sample_start > test_max_time:
                    embargo_window = (test_max_time - test_min_time) * self.embargo_pct / 0.1
                    if sample_start < test_max_time + embargo_window:
                        continue
                
                train_indices.append(i)
            
            train_indices = np.array(train_indices, dtype=int)
            
            if len(train_indices) == 0:
                logger.warning(f"Combination {combo_id}: No training samples after purging!")
                continue
            
            yield train_indices, test_indices, combo_id
    
    def get_backtest_paths(self, scores: Dict[int, float]) -> List[List[float]]:
        """
        Reconstruct backtest paths from CPCV scores.
        
        Each path represents a complete out-of-sample backtest by combining
        test fold results from different combinations.
        
        Args:
            scores: Dictionary mapping (combination_id, fold_idx) -> score
            
        Returns:
            List of backtest paths, each path is a list of scores
        """
        # Group scores by original fold index
        fold_scores = defaultdict(list)
        
        for (combo_id, fold_idx), score in scores.items():
            fold_scores[fold_idx].append(score)
        
        # Average scores per fold across all combinations where it was in test set
        avg_fold_scores = {fold: np.mean(scores_list) for fold, scores_list in fold_scores.items()}
        
        # Create paths by combining consecutive folds
        n_paths = self.n_combinations // (self.n_splits - self.n_test_folds + 1)
        paths = []
        
        for path_idx in range(min(n_paths, len(avg_fold_scores))):
            path = []
            for fold_idx in sorted(avg_fold_scores.keys()):
                if fold_idx % len(avg_fold_scores) == path_idx % len(avg_fold_scores):
                    path.append(avg_fold_scores[fold_idx])
            if path:
                paths.append(path)
        
        return paths
    
    def calculate_sharpe_distribution(self, returns: Dict[int, np.ndarray]) -> Dict[str, float]:
        """
        Calculate distribution of Sharpe ratios across backtest paths.
        
        Args:
            returns: Dictionary mapping fold_idx -> array of returns for that fold
            
        Returns:
            Statistics of Sharpe ratio distribution
        """
        sharpe_ratios = []
        
        # Calculate Sharpe for each possible path
        for combo_id, (train_idx, test_idx, _) in enumerate(self.split(np.zeros(sum(len(v) for v in returns.values())))):
            # Get returns for test indices
            test_returns = []
            for idx in test_idx:
                # Find which fold this belongs to
                for fold_idx, fold_returns in returns.items():
                    if idx < len(fold_returns):
                        test_returns.append(fold_returns[idx])
            
            if len(test_returns) > 1:
                mean_return = np.mean(test_returns)
                std_return = np.std(test_returns)
                if std_return > 0:
                    sharpe = mean_return / std_return * np.sqrt(252)  # Annualized
                    sharpe_ratios.append(sharpe)
        
        if not sharpe_ratios:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0}
        
        sharpe_array = np.array(sharpe_ratios)
        
        return {
            "mean": float(np.mean(sharpe_array)),
            "std": float(np.std(sharpe_array)),
            "min": float(np.min(sharpe_array)),
            "max": float(np.max(sharpe_array)),
            "median": float(np.median(sharpe_array)),
            "sharpe_se": float(sem(sharpe_array)) if len(sharpe_array) > 1 else 0,
            "n_paths": len(sharpe_ratios)
        }


def generate_cpcv_paths(cv_results: List[Dict[str, Any]], 
                        n_splits: int,
                        n_test_folds: int) -> List[Dict[str, Any]]:
    """
    Generate complete backtest paths from CPCV results.
    
    Args:
        cv_results: List of result dicts from each CV split
        n_splits: Number of original splits
        n_test_folds: Number of test folds per split
        
    Returns:
        List of reconstructed backtest paths
    """
    # Organize results by fold
    fold_results = defaultdict(list)
    
    for result in cv_results:
        fold_idx = result.get('fold_idx', 0)
        fold_results[fold_idx].append(result)
    
    # Create paths
    paths = []
    n_complete_paths = n_splits - n_test_folds + 1
    
    for path_idx in range(n_complete_paths):
        path = []
        for fold_idx in range(n_splits):
            # Select result from appropriate combination
            result_idx = path_idx % len(fold_results[fold_idx])
            path.append(fold_results[fold_idx][result_idx])
        paths.append(path)
    
    return paths


if __name__ == "__main__":
    # Test CPCV
    np.random.seed(42)
    
    n_samples = 1200
    X = np.random.randn(n_samples, 10)
    y = np.random.randint(0, 2, n_samples)
    
    # Create samples info with varying durations
    samples_info = {}
    for i in range(n_samples):
        duration = np.random.exponential(3)
        samples_info[i] = (float(i), float(i + duration))
    
    # Create CPCV
    cpcv = CombinatorialPurgedKFold(
        n_splits=6,
        n_test_folds=2,
        embargo_pct=0.02,
        samples_info=samples_info
    )
    
    print(f"CPCV Configuration:")
    print(f"  Total splits: {cpcv.n_splits}")
    print(f"  Test folds per split: {cpcv.n_test_folds}")
    print(f"  Total combinations: {cpcv.n_combinations}")
    print(f"  Embargo: {cpcv.embargo_pct:.2%}")
    print()
    
    # Iterate through combinations
    results = []
    for train_idx, test_idx, combo_id in cpcv.split(X):
        # Simulate model training and scoring
        train_score = np.random.randn() * 0.1 + 0.5
        test_score = np.random.randn() * 0.1 + 0.5
        
        results.append({
            'combination_id': combo_id,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'train_score': train_score,
            'test_score': test_score,
            'gap_size': n_samples - len(train_idx) - len(test_idx)
        })
        
        if combo_id < 3:  # Print first few
            print(f"Combination {combo_id}:")
            print(f"  Train: {len(train_idx)} ({len(train_idx)/n_samples:.1%})")
            print(f"  Test:  {len(test_idx)} ({len(test_idx)/n_samples:.1%})")
            print(f"  Gap:   {n_samples - len(train_idx) - len(test_idx)} purged")
            print()
    
    # Calculate Sharpe distribution from simulated returns
    print("\n--- Sharpe Ratio Distribution ---\n")
    
    # Simulate returns per fold
    fold_returns = {}
    for i in range(cpcv.n_splits):
        fold_returns[i] = np.random.randn(200) * 0.02 + 0.001
    
    sharpe_stats = cpcv.calculate_sharpe_distribution(fold_returns)
    
    print(f"Sharpe Ratio Statistics across {sharpe_stats['n_paths']} paths:")
    print(f"  Mean:   {sharpe_stats['mean']:.4f}")
    print(f"  Std:    {sharpe_stats['std']:.4f}")
    print(f"  Min:    {sharpe_stats['min']:.4f}")
    print(f"  Max:    {sharpe_stats['max']:.4f}")
    print(f"  Median: {sharpe_stats['median']:.4f}")
    print(f"  SE:     {sharpe_stats['sharpe_se']:.4f}")
    print(f"\n95% CI: [{sharpe_stats['mean'] - 1.96*sharpe_stats['std']:.4f}, "
          f"{sharpe_stats['mean'] + 1.96*sharpe_stats['std']:.4f}]")
