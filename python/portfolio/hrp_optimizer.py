"""
Hierarchical Risk Parity (HRP) Optimizer using Ray for parallel dendrogram clustering.
Implements recursive bisection without heavy matrix inversions, strictly bounding memory.
Designed for 3GB RAM limit - no Pandas in hot-path.
"""

import numpy as np
import ray
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import heapq


@ray.remote(max_calls=1000, memory=100 * 1024 * 1024)
class CorrelationClusterActor:
    """Ray actor for parallel correlation matrix computation and clustering."""
    
    def __init__(self, asset_ids: List[str]):
        self.asset_ids = asset_ids
        self.n_assets = len(asset_ids)
        self._corr_cache: Optional[np.ndarray] = None
    
    def compute_correlation_matrix(self, returns: np.ndarray) -> np.ndarray:
        """Compute correlation matrix from returns using pure NumPy."""
        if returns.shape[0] != self.n_assets:
            raise ValueError("Returns shape mismatch")
        
        # Standardize returns
        means = np.mean(returns, axis=1, keepdims=True)
        stds = np.std(returns, axis=1, keepdims=True) + 1e-10
        standardized = (returns - means) / stds
        
        # Compute correlation matrix
        corr = np.dot(standardized, standardized.T) / (returns.shape[1] - 1)
        corr = np.clip(corr, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)
        
        self._corr_cache = corr
        return corr
    
    def get_linkage_matrix(self, corr: np.ndarray) -> np.ndarray:
        """Compute linkage matrix using single-linkage hierarchical clustering."""
        n = corr.shape[0]
        distance = np.sqrt((1 - corr) / 2)
        
        # Initialize clusters
        clusters = {i: [i] for i in range(n)}
        active = set(range(n))
        linkage = []
        
        # Priority queue for distances
        heap = []
        for i in range(n):
            for j in range(i + 1, n):
                heapq.heappush(heap, (distance[i, j], i, j))
        
        cluster_id = n
        while len(active) > 1 and heap:
            dist, i, j = heapq.heappop(heap)
            
            if i not in active or j not in active:
                continue
            
            # Merge clusters
            merged = clusters[i] + clusters[j]
            linkage.append([i, j, dist, len(merged)])
            
            active.remove(i)
            active.remove(j)
            active.add(cluster_id)
            clusters[cluster_id] = merged
            
            cluster_id += 1
        
        return np.array(linkage) if linkage else np.zeros((0, 4))


@ray.remote(max_calls=1000, memory=50 * 1024 * 1024)
class RecursiveBisectionActor:
    """Ray actor for recursive bisection of the dendrogram."""
    
    def __init__(self):
        pass
    
    def bisect(self, linkage: np.ndarray, n_assets: int) -> List[int]:
        """Perform recursive bisection to get leaf ordering."""
        # Build tree structure
        children_left = linkage[:, 0].astype(int)
        children_right = linkage[:, 1].astype(int)
        
        # Find root
        root = n_assets + len(linkage) - 1
        
        # Recursive traversal
        def traverse(node: int) -> List[int]:
            if node < n_assets:
                return [node]
            
            idx = node - n_assets
            if idx >= len(linkage):
                return [node - n_assets] if node >= n_assets else [node]
            
            left = int(children_left[idx])
            right = int(children_right[idx])
            
            return traverse(left) + traverse(right)
        
        return traverse(root)
    
    def compute_cluster_variances(self, returns: np.ndarray, 
                                   leaf_order: List[int]) -> np.ndarray:
        """Compute variance for each cluster in the hierarchy."""
        n = len(leaf_order)
        cluster_vars = np.zeros(n)
        
        # Bottom-up variance computation
        for i, idx in enumerate(leaf_order):
            cluster_vars[i] = np.var(returns[idx])
        
        return cluster_vars


class HierarchicalRiskParity:
    """
    Main HRP optimizer coordinating Ray actors for parallel computation.
    Memory-bounded implementation suitable for continuous 24/7 operation.
    """
    
    def __init__(self, asset_ids: List[str], n_workers: int = 4):
        self.asset_ids = asset_ids
        self.n_assets = len(asset_ids)
        self.n_workers = min(n_workers, max(2, self.n_assets // 2))
        
        # Initialize Ray actors
        self.cluster_actors = [
            CorrelationClusterActor.remote(asset_ids) 
            for _ in range(self.n_workers)
        ]
        self.bisection_actor = RecursiveBisectionActor.remote()
        
        # Cache for weights
        self._weights_cache: Optional[np.ndarray] = None
        self._last_returns_hash: int = 0
    
    def _compute_quasi_diagonal(self, linkage: np.ndarray) -> List[int]:
        """Reorder assets to achieve quasi-diagonal correlation matrix."""
        ray_future = self.bisection_actor.bisect.remote(linkage, self.n_assets)
        return ray.get(ray_future)
    
    def _inverse_variance_weighting(self, returns: np.ndarray, 
                                     clusters: List[List[int]]) -> np.ndarray:
        """Compute inverse variance weights within each cluster."""
        weights = np.zeros(self.n_assets)
        
        for cluster in clusters:
            if not cluster:
                continue
            
            cluster_returns = returns[cluster]
            vars_inv = 1.0 / (np.var(cluster_returns, axis=1) + 1e-10)
            total_inv_var = np.sum(vars_inv)
            
            for i, idx in enumerate(cluster):
                weights[idx] = vars_inv[i] / total_inv_var
        
        return weights
    
    def _recursive_bisection_weights(self, returns: np.ndarray,
                                      leaf_order: List[int],
                                      corr: np.ndarray) -> np.ndarray:
        """Recursive bisection algorithm for weight allocation."""
        n = len(leaf_order)
        weights = np.ones(n)
        
        # Reorder correlation matrix
        corr_ordered = corr[np.ix_(leaf_order, leaf_order)]
        
        # Build clusters iteratively
        clusters = [[i] for i in range(n)]
        
        # Work up the tree
        step = 2
        while step <= n:
            for start in range(0, n, step):
                if start + step > n:
                    break
                
                left_cluster = list(range(start, start + step // 2))
                right_cluster = list(range(start + step // 2, start + step))
                
                # Compute cluster variances
                left_vars = np.sum([np.var(returns[leaf_order[i]]) for i in left_cluster])
                right_vars = np.sum([np.var(returns[leaf_order[i]]) for i in right_cluster])
                
                # Allocation factor
                alpha = 1 - left_vars / (left_vars + right_vars + 1e-10)
                
                # Update weights
                for i in left_cluster:
                    weights[i] *= alpha
                for i in right_cluster:
                    weights[i] *= (1 - alpha)
            
            step *= 2
        
        # Map back to original indices
        final_weights = np.zeros(self.n_assets)
        for i, idx in enumerate(leaf_order):
            final_weights[self.asset_ids.index(self.asset_ids[idx])] = weights[i]
        
        return final_weights
    
    def optimize(self, returns: np.ndarray, 
                 target_vol: Optional[float] = None) -> Dict[str, float]:
        """
        Compute optimal HRP weights.
        
        Args:
            returns: Asset returns array of shape (n_assets, n_samples)
            target_vol: Optional target volatility for scaling
            
        Returns:
            Dictionary mapping asset IDs to weights
        """
        if returns.shape[0] != self.n_assets:
            raise ValueError(f"Expected {self.n_assets} assets, got {returns.shape[0]}")
        
        # Check cache
        returns_hash = hash(returns.tobytes())
        if returns_hash == self._last_returns_hash and self._weights_cache is not None:
            return dict(zip(self.asset_ids, self._weights_cache))
        
        # Distribute correlation computation across actors
        chunk_size = (returns.shape[1] + self.n_workers - 1) // self.n_workers
        futures = []
        
        for i, actor in enumerate(self.cluster_actors):
            start = i * chunk_size
            end = min(start + chunk_size, returns.shape[1])
            if start < returns.shape[1]:
                chunk = returns[:, start:end]
                futures.append(actor.compute_correlation_matrix.remote(chunk))
        
        # Aggregate partial correlations (simple average for demonstration)
        partial_corrs = ray.get(futures)
        corr = np.mean(partial_corrs, axis=0)
        corr = np.clip(corr, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)
        
        # Compute linkage on primary actor
        primary_actor = self.cluster_actors[0]
        linkage_future = primary_actor.get_linkage_matrix.remote(corr)
        linkage = ray.get(linkage_future)
        
        # Get quasi-diagonal order
        leaf_order = self._compute_quasi_diagonal(linkage)
        
        # Compute weights via recursive bisection
        weights = self._recursive_bisection_weights(returns, leaf_order, corr)
        
        # Normalize weights
        weights = weights / (np.sum(weights) + 1e-10)
        
        # Scale to target volatility if specified
        if target_vol is not None:
            port_vol = np.sqrt(np.dot(weights, np.dot(corr, weights)))
            if port_vol > 1e-10:
                weights *= target_vol / port_vol
        
        # Cache results
        self._weights_cache = weights
        self._last_returns_hash = returns_hash
        
        return dict(zip(self.asset_ids, weights))
    
    def get_nautilus_allocation(self, returns: np.ndarray,
                                 portfolio_value: float) -> List[Dict]:
        """
        Generate Nautilus-compatible allocation commands.
        
        Returns:
            List of allocation dictionaries for Nautilus portfolio manager
        """
        weights = self.optimize(returns)
        
        allocations = []
        for asset_id, weight in weights.items():
            if abs(weight) > 1e-6:
                allocations.append({
                    "instrument_id": asset_id,
                    "target_weight": float(weight),
                    "target_value": float(weight * portfolio_value),
                    "rebalance_threshold": 0.05  # 5% drift tolerance
                })
        
        return allocations
    
    def cleanup(self):
        """Clean up Ray actors to free memory."""
        for actor in self.cluster_actors:
            ray.kill(actor)
        ray.kill(self.bisection_actor)


# Convenience function for direct usage
def compute_hrp_weights(asset_ids: List[str], returns: np.ndarray,
                        target_vol: Optional[float] = None) -> Dict[str, float]:
    """
    Compute HRP weights without managing actor lifecycle.
    
    Args:
        asset_ids: List of asset identifiers
        returns: Returns matrix (n_assets x n_samples)
        target_vol: Optional target volatility
        
    Returns:
        Weight dictionary
    """
    optimizer = HierarchicalRiskParity(asset_ids)
    try:
        return optimizer.optimize(returns, target_vol)
    finally:
        optimizer.cleanup()


if __name__ == "__main__":
    # Initialize Ray with memory limits
    ray.init(
        num_cpus=4,
        _system_config={
            "max_bytes_spill": 0,  # Disable spilling for low latency
            "object_store_memory": 500 * 1024 * 1024  # 500MB limit
        }
    )
    
    # Example usage
    np.random.seed(42)
    assets = ["BTC", "ETH", "SOL"]
    returns = np.random.randn(3, 1000) * 0.02
    
    optimizer = HierarchicalRiskParity(assets)
    weights = optimizer.optimize(returns, target_vol=0.15)
    
    print("HRP Weights:")
    for asset, weight in weights.items():
        print(f"  {asset}: {weight:.4f}")
    
    allocations = optimizer.get_nautilus_allocation(returns, portfolio_value=100000)
    print("\nNautilus Allocations:")
    for alloc in allocations:
        print(f"  {alloc}")
    
    optimizer.cleanup()
    ray.shutdown()
