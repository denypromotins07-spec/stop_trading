"""
HDBSCAN clustering on normalized wallet transaction graphs.
Identifies institutional accumulation syndicates using bounded Ray actors.
Processes Rust-streamed on-chain logs to detect coordinated smart money movements.
"""

import numpy as np
from numba import njit, prange
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass
import threading
from enum import IntEnum


class WalletType(IntEnum):
    """Classified wallet types."""
    UNKNOWN = 0
    RETAIL = 1
    WHALE = 2
    INSTITUTIONAL = 3
    EXCHANGE = 4
    SYNDICATE = 5  # Coordinated group


@dataclass
class WalletProfile:
    """Profile of a wallet's behavior."""
    address: str
    total_volume: float
    transaction_count: int
    avg_tx_size: float
    tx_frequency: float  # txs per hour
    unique_counterparties: int
    net_flow_24h: float
    balance_usd: float
    
    # Normalized features for clustering
    features: np.ndarray = None
    
    def __post_init__(self):
        if self.features is None:
            self.features = self._compute_features()
    
    def _compute_features(self) -> np.ndarray:
        """Compute normalized feature vector for clustering."""
        # Log-transform and normalize features
        log_volume = np.log1p(abs(self.total_volume))
        log_tx_count = np.log1p(self.transaction_count)
        log_avg_size = np.log1p(abs(self.avg_tx_size))
        log_freq = np.log1p(self.tx_frequency)
        log_counterparties = np.log1p(self.unique_counterparties)
        log_balance = np.log1p(abs(self.balance_usd))
        
        return np.array([
            log_volume,
            log_tx_count,
            log_avg_size,
            log_freq,
            log_counterparties,
            log_balance
        ], dtype=np.float32)


@njit(cache=True)
def compute_distance_matrix(data: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix."""
    n = data.shape[0]
    dist_matrix = np.zeros((n, n), dtype=np.float32)
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = data[i] - data[j]
            dist = np.sqrt(np.sum(diff ** 2))
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    return dist_matrix


@njit(cache=True)
def compute_core_distances(dist_matrix: np.ndarray, min_samples: int) -> np.ndarray:
    """Compute k-distance for each point (k = min_samples)."""
    n = dist_matrix.shape[0]
    core_dists = np.zeros(n, dtype=np.float32)
    
    for i in range(n):
        # Sort distances for point i
        sorted_dists = np.sort(dist_matrix[i])
        
        # Get k-th nearest neighbor distance
        k = min(min_samples, n - 1)
        core_dists[i] = sorted_dists[k]
    
    return core_dists


@njit(cache=True)
def compute_mutual_reachability(
    dist_matrix: np.ndarray,
    core_dists: np.ndarray,
    min_samples: int
) -> np.ndarray:
    """Compute mutual reachability distance matrix."""
    n = dist_matrix.shape[0]
    mrd_matrix = np.zeros((n, n), dtype=np.float32)
    
    for i in range(n):
        for j in range(i + 1, n):
            # MRD = max(core_dist_i, core_dist_j, dist_ij)
            mrd = max(core_dists[i], core_dists[j], dist_matrix[i, j])
            mrd_matrix[i, j] = mrd
            mrd_matrix[j, i] = mrd
    
    return mrd_matrix


@njit(cache=True)
def single_linkage_clustering(
    mrd_matrix: np.ndarray,
    min_cluster_size: int
) -> np.ndarray:
    """Simplified single-linkage hierarchical clustering."""
    n = mrd_matrix.shape[0]
    labels = np.full(n, -1, dtype=np.int32)  # -1 = noise
    
    # Union-Find structure
    parent = np.arange(n, dtype=np.int32)
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Sort all edges by weight
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((mrd_matrix[i, j], i, j))
    
    # Simple bubble sort for small datasets (use argsort for larger)
    edges.sort(key=lambda x: x[0])
    
    # Build clusters
    current_label = 0
    cluster_sizes = {}
    
    for weight, i, j in edges:
        root_i, root_j = find(i), find(j)
        
        if root_i != root_j:
            union(root_i, root_j)
    
    # Assign labels based on connected components
    label_map = {}
    for i in range(n):
        root = find(i)
        if root not in label_map:
            label_map[root] = len(label_map)
        labels[i] = label_map[root]
    
    # Count cluster sizes
    for label in labels:
        if label >= 0:
            cluster_sizes[label] = cluster_sizes.get(label, 0) + 1
    
    # Mark small clusters as noise
    for i in range(n):
        if labels[i] >= 0 and cluster_sizes[labels[i]] < min_cluster_size:
            labels[i] = -1
    
    return labels


@dataclass
class ClusterResult:
    """Result of whale clustering."""
    labels: np.ndarray
    n_clusters: int
    cluster_sizes: Dict[int, int]
    noise_count: int
    wallet_types: List[WalletType]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_clusters": self.n_clusters,
            "cluster_sizes": {str(k): v for k, v in self.cluster_sizes.items()},
            "noise_count": self.noise_count,
            "wallet_type_counts": self._count_wallet_types()
        }
    
    def _count_wallet_types(self) -> Dict[str, int]:
        counts = {}
        for wt in WalletType:
            counts[wt.name] = sum(1 for t in self.wallet_types if t == wt)
        return counts


class WhaleClustering:
    """
    HDBSCAN-inspired clustering for wallet classification.
    Detects institutional syndicates and coordinated movements.
    """
    
    def __init__(
        self,
        min_samples: int = 5,
        min_cluster_size: int = 3,
        max_cluster_size: int = 50
    ):
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        
        # State
        self._wallets: Dict[str, WalletProfile] = {}
        self._labels: Optional[np.ndarray] = None
        self._wallet_addresses: List[str] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Syndicate tracking
        self._syndicate_clusters: List[int] = []
    
    def add_wallet(self, profile: WalletProfile) -> None:
        """Add or update a wallet profile."""
        with self._lock:
            self._wallets[profile.address] = profile
            
            # Rebuild address list
            self._wallet_addresses = list(self._wallets.keys())
    
    def add_wallets_batch(self, profiles: List[WalletProfile]) -> None:
        """Add multiple wallets efficiently."""
        with self._lock:
            for profile in profiles:
                self._wallets[profile.address] = profile
            self._wallet_addresses = list(self._wallets.keys())
    
    def cluster(self) -> ClusterResult:
        """Run clustering on current wallet set."""
        with self._lock:
            if len(self._wallets) < self.min_cluster_size:
                # Not enough data
                n = len(self._wallets)
                return ClusterResult(
                    labels=np.full(n, -1, dtype=np.int32),
                    n_clusters=0,
                    cluster_sizes={},
                    noise_count=n,
                    wallet_types=[WalletType.UNKNOWN] * n
                )
            
            # Build feature matrix
            addresses = self._wallet_addresses
            n = len(addresses)
            features = np.zeros((n, 6), dtype=np.float32)
            
            for i, addr in enumerate(addresses):
                features[i] = self._wallets[addr].features
            
            # Standardize features
            means = np.mean(features, axis=0)
            stds = np.std(features, axis=0) + 1e-8
            features_norm = (features - means) / stds
            
            # Compute distance matrix
            dist_matrix = compute_distance_matrix(features_norm)
            
            # Compute core distances
            core_dists = compute_core_distances(dist_matrix, self.min_samples)
            
            # Compute mutual reachability
            mrd_matrix = compute_mutual_reachability(
                dist_matrix, core_dists, self.min_samples
            )
            
            # Cluster
            labels = single_linkage_clustering(mrd_matrix, self.min_cluster_size)
            
            # Analyze clusters
            cluster_sizes = {}
            for label in labels:
                if label >= 0:
                    cluster_sizes[label] = cluster_sizes.get(label, 0) + 1
            
            # Identify syndicate clusters (medium-sized, high activity)
            self._syndicate_clusters = []
            for cluster_id, size in cluster_sizes.items():
                if self.min_cluster_size <= size <= self.max_cluster_size:
                    self._syndicate_clusters.append(cluster_id)
            
            # Classify wallets
            wallet_types = self._classify_wallets(labels, cluster_sizes)
            
            self._labels = labels
            
            return ClusterResult(
                labels=labels.copy(),
                n_clusters=len(cluster_sizes),
                cluster_sizes=cluster_sizes,
                noise_count=sum(1 for l in labels if l < 0),
                wallet_types=wallet_types
            )
    
    def _classify_wallets(
        self,
        labels: np.ndarray,
        cluster_sizes: Dict[int, int]
    ) -> List[WalletType]:
        """Classify wallets based on clustering results and profiles."""
        wallet_types = []
        
        for i, label in enumerate(labels):
            addr = self._wallet_addresses[i]
            profile = self._wallets[addr]
            
            # Default classification
            wtype = WalletType.RETAIL
            
            # Check cluster membership
            if label >= 0:
                size = cluster_sizes.get(label, 0)
                
                if label in self._syndicate_clusters:
                    wtype = WalletType.SYNDICATE
                elif size > self.max_cluster_size:
                    # Large cluster likely exchange
                    wtype = WalletType.EXCHANGE
                else:
                    # Medium cluster - check volume
                    if profile.total_volume > 1e7:
                        wtype = WalletType.INSTITUTIONAL
                    elif profile.total_volume > 1e6:
                        wtype = WalletType.WHALE
            else:
                # Noise point - classify individually
                if profile.balance_usd > 1e8:
                    wtype = WalletType.WHALE
                elif profile.balance_usd > 1e7:
                    wtype = WalletType.INSTITUTIONAL
                elif profile.transaction_count < 10:
                    wtype = WalletType.RETAIL
            
            wallet_types.append(wtype)
        
        return wallet_types
    
    def get_syndicate_wallets(self) -> List[str]:
        """Get addresses classified as syndicate members."""
        with self._lock:
            if self._labels is None:
                return []
            
            syndicate_addrs = []
            for i, label in enumerate(self._labels):
                if label in self._syndicate_clusters:
                    syndicate_addrs.append(self._wallet_addresses[i])
            
            return syndicate_addrs
    
    def get_institutional_flow(self) -> float:
        """Calculate net flow from institutional wallets."""
        with self._lock:
            if self._labels is None:
                return 0.0
            
            total_flow = 0.0
            for i, label in enumerate(self._labels):
                addr = self._wallet_addresses[i]
                profile = self._wallets[addr]
                
                # Include institutional and syndicate wallets
                if label in self._syndicate_clusters:
                    total_flow += profile.net_flow_24h
                elif profile.balance_usd > 1e7:
                    total_flow += profile.net_flow_24h
            
            return total_flow
    
    def reset(self) -> None:
        """Reset all state."""
        with self._lock:
            self._wallets.clear()
            self._wallet_addresses.clear()
            self._labels = None
            self._syndicate_clusters.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get clustering statistics."""
        with self._lock:
            return {
                "total_wallets": len(self._wallets),
                "labeled_wallets": len(self._wallet_addresses),
                "syndicate_clusters": len(self._syndicate_clusters),
                "min_samples": self.min_samples,
                "min_cluster_size": self.min_cluster_size
            }


# Global singleton instance
_whale_instance: Optional[WhaleClustering] = None
_instance_lock = threading.Lock()


def get_whale_clusterer() -> WhaleClustering:
    """Get or create the global whale clusterer."""
    global _whale_instance
    if _whale_instance is None:
        with _instance_lock:
            if _whale_instance is None:
                _whale_instance = WhaleClustering()
    return _whale_instance


if __name__ == "__main__":
    # Test whale clustering
    print("Testing WhaleClustering:")
    
    clusterer = WhaleClustering(min_samples=3, min_cluster_size=2)
    
    np.random.seed(42)
    
    # Generate synthetic wallet data
    wallets = []
    
    # Retail wallets (small, infrequent)
    for i in range(20):
        wallets.append(WalletProfile(
            address=f"retail_{i}",
            total_volume=np.random.uniform(1e3, 1e5),
            transaction_count=np.random.randint(1, 50),
            avg_tx_size=np.random.uniform(100, 1000),
            tx_frequency=np.random.uniform(0.1, 2),
            unique_counterparties=np.random.randint(1, 10),
            net_flow_24h=np.random.uniform(-1e4, 1e4),
            balance_usd=np.random.uniform(1e3, 1e5)
        ))
    
    # Whale wallets (large volume)
    for i in range(5):
        wallets.append(WalletProfile(
            address=f"whale_{i}",
            total_volume=np.random.uniform(1e7, 1e9),
            transaction_count=np.random.randint(100, 500),
            avg_tx_size=np.random.uniform(1e5, 1e7),
            tx_frequency=np.random.uniform(5, 20),
            unique_counterparties=np.random.randint(20, 100),
            net_flow_24h=np.random.uniform(-1e6, 1e7),
            balance_usd=np.random.uniform(1e7, 1e9)
        ))
    
    # Syndicate (coordinated group)
    base_time = 1000.0
    for i in range(4):
        wallets.append(WalletProfile(
            address=f"syndicate_{i}",
            total_volume=np.random.uniform(1e6, 5e7),
            transaction_count=np.random.randint(50, 150),
            avg_tx_size=np.random.uniform(1e4, 1e6),
            tx_frequency=np.random.uniform(3, 8),
            unique_counterparties=np.random.randint(5, 20),
            net_flow_24h=np.random.uniform(1e5, 5e6),
            balance_usd=np.random.uniform(1e6, 1e8)
        ))
    
    # Add to clusterer
    clusterer.add_wallets_batch(wallets)
    
    # Run clustering
    result = clusterer.cluster()
    
    print(f"\nClustering Results:")
    print(f"  Total Clusters: {result.n_clusters}")
    print(f"  Noise Points: {result.noise_count}")
    print(f"  Wallet Types: {result._count_wallet_types()}")
    print(f"  Syndicate Wallets: {clusterer.get_syndicate_wallets()}")
    print(f"  Institutional Flow: ${clusterer.get_institutional_flow():,.2f}")
    print(f"\nStats: {clusterer.get_stats()}")
