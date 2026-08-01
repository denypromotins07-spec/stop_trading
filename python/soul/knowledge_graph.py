"""
Lightweight Knowledge Graph for SOUL.md Semantic Memory.
Uses networkx with sparse adjacency matrices to map relationships between market regimes, news events, and trade outcomes.
Strictly limits nodes and uses sparse representations to keep memory under 100MB.
No LLMs - pure graph-based memory retrieval.
"""
import asyncio
import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import time
import hashlib

try:
    import networkx as nx
    from scipy import sparse
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None
    sparse = None


class NodeType(Enum):
    MARKET_REGIME = "market_regime"
    NEWS_EVENT = "news_event"
    TRADE_OUTCOME = "trade_outcome"
    PRICE_LEVEL = "price_level"
    VOLATILITY_STATE = "volatility_state"
    LIQUIDITY_STATE = "liquidity_state"


class EdgeType(Enum):
    PRECEDES = "precedes"
    CAUSES = "causes"
    CORRELATES = "correlates"
    SIMILAR_TO = "similar_to"
    LEADS_TO = "leads_to"
    INHIBITS = "inhibits"


@dataclass
class GraphNode:
    """Node in the knowledge graph"""
    node_id: str
    node_type: NodeType
    attributes: Dict[str, Any]
    timestamp_ns: int
    importance_score: float = 1.0
    access_count: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'node_id': self.node_id,
            'node_type': self.node_type.value,
            'attributes': self.attributes,
            'timestamp_ns': self.timestamp_ns,
            'importance_score': self.importance_score,
            'access_count': self.access_count
        }


@dataclass
class GraphEdge:
    """Edge in the knowledge graph"""
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float
    timestamp_ns: int


@dataclass
class MemoryRetrieval:
    """Result of memory retrieval from graph"""
    query_node_id: str
    similar_nodes: List[GraphNode]
    related_events: List[GraphNode]
    outcome_predictions: List[Dict]
    confidence: float
    retrieval_time_ms: float


class KnowledgeGraph:
    """
    Lightweight knowledge graph using networkx with bounded memory.
    Implements sparse adjacency representation for efficiency.
    """
    
    # Memory bounds
    MAX_NODES = 5000  # Strict limit to stay under 100MB
    MAX_EDGES = 20000
    PRUNE_THRESHOLD = 0.1  # Remove nodes with importance < threshold
    
    def __init__(self):
        if not NETWORKX_AVAILABLE:
            raise ImportError("networkx is required for KnowledgeGraph")
        
        self._graph = nx.DiGraph()
        self._node_index: Dict[str, GraphNode] = {}
        self._edge_list: deque = deque(maxlen=self.MAX_EDGES)
        self._type_indices: Dict[NodeType, Set[str]] = {t: set() for t in NodeType}
        self._lock = asyncio.Lock()
        
        # Sparse adjacency cache
        self._adjacency_cache: Optional[sparse.csr_matrix] = None
        self._cache_valid = False
    
    async def add_node(self, node: GraphNode) -> bool:
        """
        Add a node to the graph with memory management.
        Returns False if graph is at capacity.
        """
        async with self._lock:
            # Check capacity
            if len(self._node_index) >= self.MAX_NODES:
                await self._prune_low_importance_nodes()
            
            # Still at capacity after pruning?
            if len(self._node_index) >= self.MAX_NODES:
                return False
            
            # Add to graph
            self._graph.add_node(
                node.node_id,
                node_type=node.node_type.value,
                **node.attributes
            )
            self._node_index[node.node_id] = node
            self._type_indices[node.node_type].add(node.node_id)
            
            self._cache_valid = False
            return True
    
    async def add_edge(self, edge: GraphEdge):
        """Add an edge to the graph"""
        async with self._lock:
            if edge.source_id not in self._node_index or edge.target_id not in self._node_index:
                return
            
            self._graph.add_edge(
                edge.source_id,
                edge.target_id,
                edge_type=edge.edge_type.value,
                weight=edge.weight
            )
            self._edge_list.append(edge)
            
            self._cache_valid = False
    
    async def _prune_low_importance_nodes(self):
        """Remove nodes with low importance scores"""
        nodes_to_remove = []
        
        for node_id, node in self._node_index.items():
            if node.importance_score < self.PRUNE_THRESHOLD:
                # Don't prune recently accessed nodes
                if time.time_ns() - node.timestamp_ns > 3600 * 1e9:  # 1 hour
                    nodes_to_remove.append(node_id)
        
        # Sort by importance and remove lowest
        nodes_to_remove.sort(key=lambda x: self._node_index[x].importance_score)
        n_remove = min(len(nodes_to_remove), self.MAX_NODES // 10)
        
        for node_id in nodes_to_remove[:n_remove]:
            await self._remove_node(node_id)
    
    async def _remove_node(self, node_id: str):
        """Remove a node from the graph"""
        if node_id not in self._node_index:
            return
        
        node = self._node_index[node_id]
        self._graph.remove_node(node_id)
        del self._node_index[node_id]
        self._type_indices[node.node_type].discard(node_id)
        
        self._cache_valid = False
    
    def update_node_importance(self, node_id: str, 
                               importance_delta: float = 0.1):
        """Update importance score for a node (called on access)"""
        if node_id in self._node_index:
            self._node_index[node_id].access_count += 1
            self._node_index[node_id].importance_score += importance_delta
            # Decay over time
            self._node_index[node_id].importance_score *= 0.99
    
    def _build_sparse_adjacency(self) -> sparse.csr_matrix:
        """Build sparse adjacency matrix for efficient operations"""
        if self._cache_valid and self._adjacency_cache is not None:
            return self._adjacency_cache
        
        n = len(self._node_index)
        if n == 0:
            return sparse.csr_matrix((0, 0))
        
        node_list = list(self._node_index.keys())
        node_to_idx = {nid: i for i, nid in enumerate(node_list)}
        
        rows = []
        cols = []
        data = []
        
        for u, v, d in self._graph.edges(data=True):
            if u in node_to_idx and v in node_to_idx:
                rows.append(node_to_idx[u])
                cols.append(node_to_idx[v])
                data.append(d.get('weight', 1.0))
        
        self._adjacency_cache = sparse.csr_matrix(
            (data, (rows, cols)), shape=(n, n)
        )
        self._cache_valid = True
        
        return self._adjacency_cache
    
    def get_neighbors(self, node_id: str, 
                      max_depth: int = 2) -> List[GraphNode]:
        """Get all nodes within max_depth hops"""
        if node_id not in self._graph:
            return []
        
        neighbors = set()
        for depth in range(1, max_depth + 1):
            for neighbor in nx.single_source_shortest_path_length(
                self._graph, node_id, cutoff=depth
            ):
                if neighbor != node_id and neighbor in self._node_index:
                    neighbors.add(neighbor)
                    self.update_node_importance(neighbor, 0.01)
        
        return [self._node_index[nid] for nid in neighbors]
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[GraphNode]:
        """Get all nodes of a specific type"""
        return [
            self._node_index[nid] 
            for nid in self._type_indices.get(node_type, set())
            if nid in self._node_index
        ]
    
    def find_similar_nodes(self, query_node: GraphNode,
                           top_k: int = 10) -> List[GraphNode]:
        """Find nodes similar to query based on attributes"""
        if not self._node_index:
            return []
        
        similarities = []
        
        for node_id, node in self._node_index.items():
            if node_id == query_node.node_id:
                continue
            if node.node_type != query_node.node_type:
                continue
            
            # Calculate attribute similarity
            sim = self._calculate_attribute_similarity(
                query_node.attributes,
                node.attributes
            )
            
            if sim > 0.3:  # Threshold
                similarities.append((node, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return [node for node, _ in similarities[:top_k]]
    
    def _calculate_attribute_similarity(self, attrs1: Dict, 
                                         attrs2: Dict) -> float:
        """Calculate similarity between two attribute dictionaries"""
        common_keys = set(attrs1.keys()) & set(attrs2.keys())
        if not common_keys:
            return 0.0
        
        total_sim = 0.0
        n_comparisons = 0
        
        for key in common_keys:
            v1, v2 = attrs1[key], attrs2[key]
            
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                # Numeric similarity (inverse of relative difference)
                if v1 == 0 and v2 == 0:
                    sim = 1.0
                else:
                    diff = abs(v1 - v2) / (max(abs(v1), abs(v2)) + 1e-10)
                    sim = 1.0 - diff
            elif isinstance(v1, str) and isinstance(v2, str):
                # String similarity
                sim = 1.0 if v1 == v2 else 0.0
            else:
                sim = 0.5 if v1 == v2 else 0.0
            
            total_sim += sim
            n_comparisons += 1
        
        return total_sim / n_comparisons if n_comparisons > 0 else 0.0
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get current graph statistics"""
        return {
            'num_nodes': len(self._node_index),
            'num_edges': self._graph.number_of_edges(),
            'nodes_by_type': {
                t.value: len(ids) for t, ids in self._type_indices.items()
            },
            'avg_degree': (2 * self._graph.number_of_edges() / 
                          max(len(self._node_index), 1)),
            'memory_estimate_mb': self._estimate_memory_mb()
        }
    
    def _estimate_memory_mb(self) -> float:
        """Estimate memory usage in MB"""
        # Rough estimate
        node_mem = len(self._node_index) * 200  # ~200 bytes per node
        edge_mem = self._graph.number_of_edges() * 100  # ~100 bytes per edge
        return (node_mem + edge_mem) / (1024 * 1024)
    
    def export_subgraph(self, seed_nodes: List[str],
                        max_depth: int = 2) -> Dict:
        """Export a subgraph for external analysis"""
        if not seed_nodes:
            return {'nodes': [], 'edges': []}
        
        # Get all nodes in subgraph
        subgraph_nodes = set(seed_nodes)
        for seed in seed_nodes:
            if seed in self._graph:
                for node in nx.single_source_shortest_path_length(
                    self._graph, seed, cutoff=max_depth
                ):
                    subgraph_nodes.add(node)
        
        # Build subgraph data
        nodes = [
            self._node_index[nid].to_dict()
            for nid in subgraph_nodes
            if nid in self._node_index
        ]
        
        edges = [
            {
                'source': u,
                'target': v,
                'type': d.get('edge_type', 'unknown'),
                'weight': d.get('weight', 1.0)
            }
            for u, v, d in self._graph.edges(data=True)
            if u in subgraph_nodes and v in subgraph_nodes
        ]
        
        return {'nodes': nodes, 'edges': edges}


# Global singleton instance
_graph_instance: Optional[KnowledgeGraph] = None


def get_knowledge_graph() -> KnowledgeGraph:
    """Get or create global knowledge graph"""
    global _graph_instance
    if _graph_instance is None:
        if not NETWORKX_AVAILABLE:
            raise RuntimeError("networkx is required but not installed")
        _graph_instance = KnowledgeGraph()
    return _graph_instance


async def demo():
    """Demo usage of the knowledge graph"""
    print("=== Knowledge Graph Demo ===\n")
    
    kg = get_knowledge_graph()
    
    base_time = time.time_ns()
    
    # Add market regime nodes
    regimes = [
        ("high_vol_bull", NodeType.MARKET_REGIME, {'volatility': 0.8, 'trend': 1.0}),
        ("low_vol_consolidation", NodeType.MARKET_REGIME, {'volatility': 0.2, 'trend': 0.0}),
        ("flash_crash", NodeType.MARKET_REGIME, {'volatility': 2.0, 'trend': -1.0}),
    ]
    
    for name, ntype, attrs in regimes:
        node = GraphNode(
            node_id=name,
            node_type=ntype,
            attributes=attrs,
            timestamp_ns=base_time
        )
        await kg.add_node(node)
    
    # Add news event nodes
    news = [
        ("fed_hike", NodeType.NEWS_EVENT, {'impact': 'negative', 'magnitude': 0.7}),
        ("etf_approval", NodeType.NEWS_EVENT, {'impact': 'positive', 'magnitude': 0.9}),
    ]
    
    for name, ntype, attrs in news:
        node = GraphNode(
            node_id=name,
            node_type=ntype,
            attributes=attrs,
            timestamp_ns=base_time
        )
        await kg.add_node(node)
    
    # Add edges
    edges = [
        GraphEdge("fed_hike", "high_vol_bull", EdgeType.CAUSES, 0.8, base_time),
        GraphEdge("etf_approval", "high_vol_bull", EdgeType.CAUSES, 0.6, base_time),
        GraphEdge("high_vol_bull", "flash_crash", EdgeType.PRECEDES, 0.3, base_time),
    ]
    
    for edge in edges:
        await kg.add_edge(edge)
    
    # Get stats
    stats = kg.get_graph_stats()
    print(f"Graph Stats:")
    print(f"  Nodes: {stats['num_nodes']}")
    print(f"  Edges: {stats['num_edges']}")
    print(f"  Est. Memory: {stats['memory_estimate_mb']:.2f} MB")
    
    # Find similar nodes
    query = GraphNode(
        node_id="query",
        node_type=NodeType.MARKET_REGIME,
        attributes={'volatility': 0.75, 'trend': 0.9},
        timestamp_ns=base_time
    )
    
    similar = kg.find_similar_nodes(query, top_k=3)
    print(f"\nSimilar nodes to query:")
    for node in similar:
        print(f"  - {node.node_id} (type: {node.node_type.value})")
    
    # Get neighbors
    neighbors = kg.get_neighbors("fed_hike", max_depth=2)
    print(f"\nNeighbors of 'fed_hike': {[n.node_id for n in neighbors]}")


if __name__ == "__main__":
    asyncio.run(demo())
