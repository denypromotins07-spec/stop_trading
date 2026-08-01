"""
Louvain community detection algorithm using networkx to identify coordinated whale syndicates
and exchange hot wallets. Strictly prunes stale nodes and edges to keep memory footprint under 200MB.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
import logging
import time
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class GraphStats:
    """Statistics about the transaction graph."""
    num_nodes: int = 0
    num_edges: int = 0
    num_communities: int = 0
    avg_degree: float = 0.0
    max_degree: int = 0
    density: float = 0.0
    memory_mb: float = 0.0
    last_updated_ns: int = field(default_factory=lambda: time.time_ns())


@dataclass
class CommunityInfo:
    """Information about a detected community."""
    community_id: int
    members: Set[str]
    size: int
    total_volume: float
    avg_tx_value: float
    internal_edge_count: int
    external_edge_count: int
    modularity_contribution: float
    
    # Classification hints
    is_whale_syndicate: bool = False
    is_exchange_cluster: bool = False
    risk_score: float = 0.0


class BoundedTransactionGraph:
    """
    Transaction graph with bounded memory footprint.
    Automatically prunes stale nodes and edges to stay under memory limits.
    """
    
    def __init__(
        self,
        max_memory_mb: float = 200.0,
        stale_threshold_seconds: float = 3600.0,  # 1 hour
        max_nodes: int = 100000
    ):
        self.max_memory_mb = max_memory_mb
        self.stale_threshold_seconds = stale_threshold_seconds
        self.max_nodes = max_nodes
        
        self.graph = nx.DiGraph()
        
        # Track node timestamps for pruning
        self._node_timestamps: Dict[str, int] = {}
        self._edge_timestamps: Dict[Tuple[str, str], int] = {}
        
        # Node attributes cache
        self._node_volumes: Dict[str, float] = {}
        self._node_degrees: Dict[str, int] = {}
        
        logger.info(f"BoundedTransactionGraph initialized (max {max_memory_mb}MB)")
    
    def add_transaction(
        self,
        from_addr: str,
        to_addr: str,
        value: float,
        timestamp_ns: Optional[int] = None
    ) -> bool:
        """
        Add a transaction edge to the graph.
        Returns False if operation would exceed memory limits.
        """
        current_time = timestamp_ns or time.time_ns()
        
        # Check if we need to prune before adding
        if not self._can_add_node():
            self._prune_stale_nodes()
        
        # Add/update nodes
        for addr in [from_addr, to_addr]:
            if addr not in self.graph:
                if not self._can_add_node():
                    logger.warning("Graph at capacity, dropping transaction")
                    return False
                self.graph.add_node(addr)
                self._node_timestamps[addr] = current_time
                self._node_volumes[addr] = 0.0
                self._node_degrees[addr] = 0
        
        # Add edge
        edge_key = (from_addr, to_addr)
        self.graph.add_edge(from_addr, to_addr, weight=value, timestamp=current_time)
        self._edge_timestamps[edge_key] = current_time
        
        # Update node attributes
        self._node_volumes[from_addr] = self._node_volumes.get(from_addr, 0.0) + value
        self._node_volumes[to_addr] = self._node_volumes.get(to_addr, 0.0) + value
        self._node_degrees[from_addr] = self._node_degrees.get(from_addr, 0) + 1
        self._node_degrees[to_addr] = self._node_degrees.get(to_addr, 0) + 1
        
        # Update node timestamps
        self._node_timestamps[from_addr] = current_time
        self._node_timestamps[to_addr] = current_time
        
        return True
    
    def _can_add_node(self) -> bool:
        """Check if we can add another node without exceeding limits."""
        current_nodes = len(self.graph.nodes())
        current_memory = self._estimate_memory_mb()
        
        return (current_nodes < self.max_nodes and 
                current_memory < self.max_memory_mb * 0.95)
    
    def _estimate_memory_mb(self) -> float:
        """Estimate current memory usage in MB."""
        # Rough estimation based on node/edge counts
        num_nodes = len(self.graph.nodes())
        num_edges = len(self.graph.edges())
        
        # Approximate bytes per element (very rough)
        node_bytes = num_nodes * 200  # Address string + attributes
        edge_bytes = num_edges * 100  # Edge attributes
        
        total_bytes = node_bytes + edge_bytes
        return total_bytes / (1024 * 1024)
    
    def _prune_stale_nodes(self):
        """Remove stale nodes and their edges."""
        current_time = time.time_ns()
        threshold_ns = int(self.stale_threshold_seconds * 1e9)
        
        nodes_to_remove = []
        
        for node in list(self.graph.nodes()):
            node_time = self._node_timestamps.get(node, 0)
            if current_time - node_time > threshold_ns:
                # Check if node has recent transactions
                has_recent = False
                for neighbor in self.graph.neighbors(node):
                    edge_key = (node, neighbor) if self.graph.has_edge(node, neighbor) else (neighbor, node)
                    edge_time = self._edge_timestamps.get(edge_key, 0)
                    if current_time - edge_time < threshold_ns:
                        has_recent = True
                        break
                
                if not has_recent:
                    nodes_to_remove.append(node)
        
        # Remove stale nodes
        for node in nodes_to_remove:
            self.graph.remove_node(node)
            self._node_timestamps.pop(node, None)
            self._node_volumes.pop(node, None)
            self._node_degrees.pop(node, None)
        
        # Clean up edge timestamps for removed edges
        edges_to_remove = [k for k in self._edge_timestamps.keys() 
                          if k[0] not in self.graph or k[1] not in self.graph]
        for edge_key in edges_to_remove:
            self._edge_timestamps.pop(edge_key, None)
        
        if nodes_to_remove:
            logger.debug(f"Pruned {len(nodes_to_remove)} stale nodes")
    
    def get_stats(self) -> GraphStats:
        """Get current graph statistics."""
        num_nodes = len(self.graph.nodes())
        num_edges = len(self.graph.edges())
        
        degrees = [d for n, d in self.graph.degree()]
        avg_degree = np.mean(degrees) if degrees else 0.0
        max_degree = max(degrees) if degrees else 0
        
        # Density for directed graph
        max_edges = num_nodes * (num_nodes - 1)
        density = num_edges / max_edges if max_edges > 0 else 0.0
        
        return GraphStats(
            num_nodes=num_nodes,
            num_edges=num_edges,
            avg_degree=avg_degree,
            max_degree=max_degree,
            density=density,
            memory_mb=self._estimate_memory_mb(),
            last_updated_ns=time.time_ns()
        )


class LouvainCommunityDetector:
    """
    Louvain community detection for transaction graphs.
    Identifies whale syndicates and exchange clusters.
    """
    
    def __init__(
        self,
        graph: BoundedTransactionGraph,
        resolution: float = 1.0,
        min_community_size: int = 5
    ):
        self.graph = graph
        self.resolution = resolution
        self.min_community_size = min_community_size
        
        self._communities: List[CommunityInfo] = []
        self._node_to_community: Dict[str, int] = {}
        self._last_detection_time: int = 0
    
    def detect_communities(self) -> List[CommunityInfo]:
        """
        Run Louvain community detection on the transaction graph.
        """
        start_time = time.perf_counter()
        
        # Convert to undirected for community detection
        undirected = self.graph.graph.to_undirected()
        
        if len(undirected.nodes()) < self.min_community_size:
            logger.warning("Graph too small for community detection")
            return []
        
        # Run Louvain algorithm
        try:
            partition = nx.community.louvain_communities(
                undirected,
                resolution=self.resolution,
                seed=42
            )
        except Exception as e:
            logger.error(f"Louvain detection failed: {e}")
            return []
        
        # Process communities
        communities = []
        for i, member_set in enumerate(partition):
            if len(member_set) < self.min_community_size:
                continue
            
            community_info = self._analyze_community(i, set(member_set))
            if community_info:
                communities.append(community_info)
                
                # Map nodes to community
                for node in member_set:
                    self._node_to_community[node] = i
        
        self._communities = communities
        self._last_detection_time = time.time_ns()
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"Detected {len(communities)} communities in {elapsed_ms:.1f}ms")
        
        return communities
    
    def _analyze_community(
        self,
        community_id: int,
        members: Set[str]
    ) -> Optional[CommunityInfo]:
        """Analyze a community and compute metrics."""
        if not members:
            return None
        
        # Calculate volume metrics
        total_volume = sum(
            self.graph._node_volumes.get(node, 0.0) for node in members
        )
        
        # Count internal vs external edges
        internal_edges = 0
        external_edges = 0
        
        for node in members:
            for neighbor in self.graph.graph.neighbors(node):
                if neighbor in members:
                    internal_edges += 1
                else:
                    external_edges += 1
        
        # Average transaction value
        tx_values = []
        for u, v, data in self.graph.graph.edges(members, data=True):
            if 'weight' in data:
                tx_values.append(data['weight'])
        
        avg_tx_value = np.mean(tx_values) if tx_values else 0.0
        
        # Classify community
        is_whale_syndicate = self._is_whale_syndicate(members, total_volume, internal_edges)
        is_exchange_cluster = self._is_exchange_cluster(members, external_edges)
        
        # Compute risk score
        risk_score = self._compute_risk_score(
            members, total_volume, is_whale_syndicate, is_exchange_cluster
        )
        
        # Modularity contribution (simplified)
        modularity_contrib = internal_edges / (internal_edges + external_edges + 1)
        
        return CommunityInfo(
            community_id=community_id,
            members=members,
            size=len(members),
            total_volume=total_volume,
            avg_tx_value=avg_tx_value,
            internal_edge_count=internal_edges,
            external_edge_count=external_edges,
            modularity_contribution=modularity_contrib,
            is_whale_syndicate=is_whale_syndicate,
            is_exchange_cluster=is_exchange_cluster,
            risk_score=risk_score
        )
    
    def _is_whale_syndicate(
        self,
        members: Set[str],
        total_volume: float,
        internal_edges: int
    ) -> bool:
        """Detect if community is a whale syndicate."""
        # High internal connectivity and large volume
        avg_volume = total_volume / len(members) if members else 0
        
        return (avg_volume > 100.0 and  # Large average volume
                internal_edges > len(members) * 2)  # High internal connectivity
    
    def _is_exchange_cluster(
        self,
        members: Set[str],
        external_edges: int
    ) -> bool:
        """Detect if community is an exchange cluster."""
        # Many external connections (customer withdrawals/deposits)
        avg_external = external_edges / len(members) if members else 0
        
        return avg_external > 10  # High external connectivity
    
    def _compute_risk_score(
        self,
        members: Set[str],
        total_volume: float,
        is_whale_syndicate: bool,
        is_exchange_cluster: bool
    ) -> float:
        """Compute risk score for the community."""
        score = 0.0
        
        # Volume-based risk
        if total_volume > 10000:
            score += 0.3
        elif total_volume > 1000:
            score += 0.15
        
        # Syndicate risk
        if is_whale_syndicate:
            score += 0.4
        
        # Exchange risk (lower, but still tracked)
        if is_exchange_cluster:
            score += 0.1
        
        # Size-based adjustment
        if len(members) > 50:
            score += 0.1
        
        return min(score, 1.0)
    
    def get_community_for_node(self, address: str) -> Optional[CommunityInfo]:
        """Get community info for a specific address."""
        community_id = self._node_to_community.get(address)
        if community_id is None:
            return None
        
        for comm in self._communities:
            if comm.community_id == community_id:
                return comm
        
        return None
    
    def get_whale_syndicates(self) -> List[CommunityInfo]:
        """Get all detected whale syndicates."""
        return [c for c in self._communities if c.is_whale_syndicate]
    
    def get_exchange_clusters(self) -> List[CommunityInfo]:
        """Get all detected exchange clusters."""
        return [c for c in self._communities if c.is_exchange_cluster]
    
    def get_high_risk_communities(self, min_risk: float = 0.5) -> List[CommunityInfo]:
        """Get communities with risk score above threshold."""
        return [c for c in self._communities if c.risk_score >= min_risk]


class OnChainGraphAnalytics:
    """
    Main class orchestrating on-chain graph analytics.
    Combines GraphSAGE classification with community detection.
    """
    
    def __init__(
        self,
        max_memory_mb: float = 200.0,
        stale_threshold_hours: float = 1.0,
        louvain_resolution: float = 1.0
    ):
        self.transaction_graph = BoundedTransactionGraph(
            max_memory_mb=max_memory_mb,
            stale_threshold_seconds=stale_threshold_hours * 3600
        )
        
        self.community_detector = LouvainCommunityDetector(
            self.transaction_graph,
            resolution=louvain_resolution
        )
        
        self._transaction_count = 0
        self._last_community_detection = 0
        self._detection_interval_transactions = 1000
    
    def process_transaction(
        self,
        from_addr: str,
        to_addr: str,
        value: float,
        timestamp_ns: Optional[int] = None
    ) -> bool:
        """Process a new transaction."""
        success = self.transaction_graph.add_transaction(
            from_addr, to_addr, value, timestamp_ns
        )
        
        if success:
            self._transaction_count += 1
            
            # Periodically run community detection
            if (self._transaction_count - self._last_community_detection >= 
                self._detection_interval_transactions):
                self.run_community_detection()
                self._last_community_detection = self._transaction_count
        
        return success
    
    def run_community_detection(self) -> List[CommunityInfo]:
        """Run community detection on current graph state."""
        return self.community_detector.detect_communities()
    
    def analyze_address(self, address: str) -> Dict[str, Any]:
        """Get comprehensive analysis for an address."""
        result = {
            'address': address,
            'in_graph': address in self.transaction_graph.graph,
            'volume': self.transaction_graph._node_volumes.get(address, 0.0),
            'degree': self.transaction_graph._node_degrees.get(address, 0),
            'community': None,
            'risk_assessment': {}
        }
        
        if result['in_graph']:
            community = self.community_detector.get_community_for_node(address)
            if community:
                result['community'] = {
                    'id': community.community_id,
                    'size': community.size,
                    'is_whale_syndicate': community.is_whale_syndicate,
                    'is_exchange_cluster': community.is_exchange_cluster,
                    'risk_score': community.risk_score
                }
                
                result['risk_assessment'] = {
                    'overall_risk': community.risk_score,
                    'syndicate_member': community.is_whale_syndicate,
                    'exchange_related': community.is_exchange_cluster,
                    'community_volume': community.total_volume
                }
        
        return result
    
    def get_graph_stats(self) -> GraphStats:
        """Get current graph statistics."""
        stats = self.transaction_graph.get_stats()
        stats.num_communities = len(self.community_detector._communities)
        return stats
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status for monitoring."""
        stats = self.get_graph_stats()
        
        return {
            'transaction_count': self._transaction_count,
            'graph_stats': {
                'nodes': stats.num_nodes,
                'edges': stats.num_edges,
                'communities': stats.num_communities,
                'memory_mb': stats.memory_mb
            },
            'whale_syndicates': len(self.community_detector.get_whale_syndicates()),
            'exchange_clusters': len(self.community_detector.get_exchange_clusters()),
            'high_risk_communities': len(self.community_detector.get_high_risk_communities(0.5))
        }


# Module singleton
_module_instance: Optional[OnChainGraphAnalytics] = None


def get_module() -> OnChainGraphAnalytics:
    """Get or create module singleton."""
    global _module_instance
    if _module_instance is None:
        _module_instance = OnChainGraphAnalytics()
    return _module_instance


def initialize_module(
    max_memory_mb: float = 200.0,
    stale_threshold_hours: float = 1.0,
    louvain_resolution: float = 1.0
) -> OnChainGraphAnalytics:
    """Initialize the module with configuration."""
    global _module_instance
    _module_instance = OnChainGraphAnalytics(
        max_memory_mb=max_memory_mb,
        stale_threshold_hours=stale_threshold_hours,
        louvain_resolution=louvain_resolution
    )
    return _module_instance
