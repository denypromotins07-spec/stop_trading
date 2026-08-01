"""
SOUL.md Knowledge Graph Module Root.
Manages the graph's lifecycle, strictly bounding node counts and pruning stale edges.
"""
from .knowledge_graph import (
    KnowledgeGraph,
    NodeType,
    EdgeType,
    GraphNode,
    GraphEdge,
    MemoryRetrieval,
    get_knowledge_graph
)
from .graph_memory import (
    GraphMemory,
    MarketState,
    HistoricalMemory,
    RetrievedContext,
    get_graph_memory
)

__all__ = [
    # Knowledge Graph
    "KnowledgeGraph",
    "NodeType",
    "EdgeType",
    "GraphNode",
    "GraphEdge",
    "MemoryRetrieval",
    "get_knowledge_graph",
    
    # Graph Memory
    "GraphMemory",
    "MarketState",
    "HistoricalMemory",
    "RetrievedContext",
    "get_graph_memory",
]
