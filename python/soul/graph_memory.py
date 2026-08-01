"""
Graph Memory for SOUL.md Semantic Retrieval.
Implements graph traversal algorithms to retrieve contextual memories when encountering similar historical states.
Allows the bot to "remember" specific market crashes or liquidity events without LLMs or vector databases.
Strictly bounded memory footprint.
"""
import asyncio
import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import time

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


@dataclass
class MarketState:
    """Current market state for similarity matching"""
    timestamp_ns: int
    volatility: float
    trend: float
    liquidity_score: float
    regime_type: str
    recent_events: List[str]


@dataclass
class HistoricalMemory:
    """A stored memory of a past market state and outcome"""
    memory_id: str
    state: MarketState
    outcome: Dict[str, Any]  # e.g., {'pnl': -0.05, 'slippage_bps': 15}
    actions_taken: List[str]
    lessons_learned: List[str]
    importance: float
    access_count: int = 0
    last_access_ns: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'memory_id': self.memory_id,
            'state': {
                'volatility': self.state.volatility,
                'trend': self.state.trend,
                'liquidity_score': self.state.liquidity_score,
                'regime_type': self.state.regime_type
            },
            'outcome': self.outcome,
            'actions_taken': self.actions_taken,
            'lessons_learned': self.lessons_learned,
            'importance': self.importance,
            'access_count': self.access_count
        }


@dataclass
class RetrievedContext:
    """Context retrieved from memory for current situation"""
    current_state: MarketState
    similar_memories: List[HistoricalMemory]
    recommended_actions: List[str]
    risk_warnings: List[str]
    confidence_score: float
    retrieval_time_ms: float


class GraphMemory:
    """
    Graph-based memory system for retrieving historical contexts.
    Uses state similarity and graph traversal to find relevant memories.
    """
    
    # Configuration
    MAX_MEMORIES = 2000  # Bounded memory
    SIMILARITY_THRESHOLD = 0.6
    MAX_RETRIEVAL_DEPTH = 3
    TOP_K_SIMILAR = 5
    
    # State weights for similarity calculation
    STATE_WEIGHTS = {
        'volatility': 0.35,
        'trend': 0.25,
        'liquidity_score': 0.25,
        'regime_match': 0.15
    }
    
    def __init__(self):
        if not NETWORKX_AVAILABLE:
            raise ImportError("networkx is required for GraphMemory")
        
        self._memories: Dict[str, HistoricalMemory] = {}
        self._graph = nx.DiGraph()
        self._regime_index: Dict[str, Set[str]] = {}
        self._lock = asyncio.Lock()
        
        # Decay parameters
        self.importance_decay = 0.995
        self.min_importance = 0.1
    
    async def store_memory(self, memory: HistoricalMemory):
        """Store a new memory in the graph"""
        async with self._lock:
            # Check capacity
            if len(self._memories) >= self.MAX_MEMORIES:
                await self._prune_old_memories()
            
            # Store memory
            self._memories[memory.memory_id] = memory
            
            # Add to graph
            self._graph.add_node(
                memory.memory_id,
                regime=memory.state.regime_type,
                volatility=memory.state.volatility,
                outcome_type=self._categorize_outcome(memory.outcome)
            )
            
            # Index by regime
            regime = memory.state.regime_type
            if regime not in self._regime_index:
                self._regime_index[regime] = set()
            self._regime_index[regime].add(memory.memory_id)
            
            # Create edges to similar existing memories
            await self._create_similarity_edges(memory)
    
    async def _create_similarity_edges(self, new_memory: HistoricalMemory):
        """Create edges between new memory and similar existing ones"""
        for mem_id, existing in self._memories.items():
            if mem_id == new_memory.memory_id:
                continue
            
            similarity = self._calculate_state_similarity(
                new_memory.state,
                existing.state
            )
            
            if similarity > self.SIMILARITY_THRESHOLD:
                self._graph.add_edge(
                    new_memory.memory_id,
                    mem_id,
                    similarity=similarity,
                    edge_type='similar_state'
                )
                
                # Limit edges per node
                if self._graph.out_degree(new_memory.memory_id) > 20:
                    break
    
    async def _prune_old_memories(self):
        """Remove low-importance memories"""
        candidates = []
        
        for mem_id, memory in self._memories.items():
            # Apply decay
            memory.importance *= self.importance_decay
            
            if memory.importance < self.min_importance:
                # Older memories are more likely to be pruned
                age_hours = (time.time_ns() - memory.last_access_ns) / 3.6e12
                prune_score = memory.importance / (1 + age_hours * 0.1)
                candidates.append((mem_id, prune_score))
        
        # Sort and remove lowest scoring
        candidates.sort(key=lambda x: x[1])
        n_remove = min(len(candidates), self.MAX_MEMORIES // 10)
        
        for mem_id, _ in candidates[:n_remove]:
            await self._remove_memory(mem_id)
    
    async def _remove_memory(self, memory_id: str):
        """Remove a memory from the system"""
        if memory_id not in self._memories:
            return
        
        memory = self._memories[memory_id]
        regime = memory.state.regime_type
        
        del self._memories[memory_id]
        self._graph.remove_node(memory_id)
        
        if regime in self._regime_index:
            self._regime_index[regime].discard(memory_id)
    
    async def retrieve_context(self, current_state: MarketState) -> RetrievedContext:
        """
        Retrieve relevant historical context for current market state.
        Uses graph traversal to find similar situations and their outcomes.
        """
        start_time = time.perf_counter()
        
        async with self._lock:
            # Find directly similar memories
            similar_memories = self._find_similar_memories(current_state)
            
            # Expand via graph traversal
            expanded_memories = await self._expand_via_graph(
                similar_memories,
                max_depth=self.MAX_RETRIEVAL_DEPTH
            )
            
            # Extract recommendations from outcomes
            recommendations = self._extract_recommendations(expanded_memories)
            
            # Identify risk warnings
            warnings = self._identify_risks(expanded_memories, current_state)
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                expanded_memories,
                current_state
            )
            
            retrieval_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Update access counts
            for memory in expanded_memories:
                memory.access_count += 1
                memory.last_access_ns = time.time_ns()
                memory.importance = min(1.0, memory.importance + 0.05)
            
            return RetrievedContext(
                current_state=current_state,
                similar_memories=expanded_memories[:self.TOP_K_SIMILAR],
                recommended_actions=recommendations,
                risk_warnings=warnings,
                confidence_score=confidence,
                retrieval_time_ms=retrieval_time_ms
            )
    
    def _find_similar_memories(self, state: MarketState) -> List[HistoricalMemory]:
        """Find memories with similar states"""
        similarities = []
        
        for mem_id, memory in self._memories.items():
            sim = self._calculate_state_similarity(state, memory.state)
            if sim > self.SIMILARITY_THRESHOLD:
                similarities.append((memory, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in similarities]
    
    def _calculate_state_similarity(self, s1: MarketState, 
                                     s2: MarketState) -> float:
        """Calculate similarity between two market states"""
        # Volatility similarity
        vol_diff = abs(s1.volatility - s2.volatility) / max(s1.volatility, s2.volatility, 0.01)
        vol_sim = 1.0 - min(vol_diff, 1.0)
        
        # Trend similarity
        trend_diff = abs(s1.trend - s2.trend) / 2.0
        trend_sim = 1.0 - trend_diff
        
        # Liquidity similarity
        liq_diff = abs(s1.liquidity_score - s2.liquidity_score)
        liq_sim = 1.0 - liq_diff
        
        # Regime match
        regime_match = 1.0 if s1.regime_type == s2.regime_type else 0.0
        
        # Weighted combination
        total_sim = (
            self.STATE_WEIGHTS['volatility'] * vol_sim +
            self.STATE_WEIGHTS['trend'] * trend_sim +
            self.STATE_WEIGHTS['liquidity_score'] * liq_sim +
            self.STATE_WEIGHTS['regime_match'] * regime_match
        )
        
        return total_sim
    
    async def _expand_via_graph(self, seed_memories: List[HistoricalMemory],
                                 max_depth: int) -> List[HistoricalMemory]:
        """Expand search via graph traversal"""
        if not seed_memories:
            return []
        
        expanded = set(m.memory_id for m in seed_memories)
        
        for memory in seed_memories:
            if memory.memory_id not in self._graph:
                continue
            
            # BFS traversal
            try:
                neighbors = nx.single_source_shortest_path_length(
                    self._graph,
                    memory.memory_id,
                    cutoff=max_depth
                )
                
                for neighbor_id in neighbors:
                    if neighbor_id != memory.memory_id and neighbor_id in self._memories:
                        expanded.add(neighbor_id)
            except Exception:
                pass
        
        return [self._memories[mid] for mid in expanded]
    
    def _extract_recommendations(self, memories: List[HistoricalMemory]) -> List[str]:
        """Extract action recommendations from memory outcomes"""
        if not memories:
            return []
        
        # Count successful vs unsuccessful actions
        action_scores: Dict[str, float] = {}
        
        for memory in memories:
            outcome_pnl = memory.outcome.get('pnl', 0)
            weight = memory.importance * (1 if outcome_pnl > 0 else 0.5)
            
            for action in memory.actions_taken:
                if action not in action_scores:
                    action_scores[action] = 0
                action_scores[action] += weight * (1 if outcome_pnl > 0 else -0.5)
        
        # Sort by score
        sorted_actions = sorted(action_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [action for action, score in sorted_actions[:5] if score > 0]
    
    def _identify_risks(self, memories: List[HistoricalMemory],
                        current_state: MarketState) -> List[str]:
        """Identify potential risks based on historical outcomes"""
        warnings = []
        
        negative_outcomes = [
            m for m in memories 
            if m.outcome.get('pnl', 0) < -0.02  # >2% loss
        ]
        
        if len(negative_outcomes) > len(memories) * 0.3:
            warnings.append("High probability of losses in similar historical states")
        
        # Check for specific risk patterns
        high_slippage = [
            m for m in memories
            if m.outcome.get('slippage_bps', 0) > 50
        ]
        
        if high_slippage and current_state.liquidity_score < 0.5:
            warnings.append("Low liquidity may cause significant slippage")
        
        # Flash crash pattern
        flash_crash_memories = [
            m for m in memories
            if 'flash_crash' in m.state.regime_type.lower()
        ]
        
        if flash_crash_memories and current_state.volatility > 1.0:
            warnings.append("High volatility regime - increased flash crash risk")
        
        return warnings
    
    def _calculate_confidence(self, memories: List[HistoricalMemory],
                               state: MarketState) -> float:
        """Calculate confidence score for recommendations"""
        if not memories:
            return 0.0
        
        # Base confidence on number of similar memories
        n_factor = min(1.0, len(memories) / 10)
        
        # Quality factor based on outcome consistency
        pnls = [m.outcome.get('pnl', 0) for m in memories]
        if len(pnls) > 1:
            pnl_std = np.std(pnls)
            quality_factor = 1.0 / (1.0 + pnl_std)
        else:
            quality_factor = 0.5
        
        # Recency factor
        now = time.time_ns()
        ages = [(now - m.last_access_ns) / 1e9 for m in memories]
        avg_age_days = np.mean(ages) / 86400 if ages else 30
        recency_factor = 1.0 / (1.0 + avg_age_days * 0.05)
        
        confidence = n_factor * 0.4 + quality_factor * 0.4 + recency_factor * 0.2
        
        return min(1.0, confidence)
    
    def _categorize_outcome(self, outcome: Dict) -> str:
        """Categorize outcome type for indexing"""
        pnl = outcome.get('pnl', 0)
        if pnl > 0.02:
            return 'highly_profitable'
        elif pnl > 0:
            return 'profitable'
        elif pnl > -0.02:
            return 'small_loss'
        else:
            return 'significant_loss'
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories"""
        if not self._memories:
            return {'num_memories': 0}
        
        regimes = {}
        for memory in self._memories.values():
            regime = memory.state.regime_type
            regimes[regime] = regimes.get(regime, 0) + 1
        
        avg_importance = np.mean([m.importance for m in self._memories.values()])
        avg_access = np.mean([m.access_count for m in self._memories.values()])
        
        return {
            'num_memories': len(self._memories),
            'num_edges': self._graph.number_of_edges(),
            'regimes': regimes,
            'avg_importance': float(avg_importance),
            'avg_access_count': float(avg_access),
            'capacity_used_pct': len(self._memories) / self.MAX_MEMORIES * 100
        }


# Global singleton instance
_memory_instance: Optional[GraphMemory] = None


def get_graph_memory() -> GraphMemory:
    """Get or create global graph memory"""
    global _memory_instance
    if _memory_instance is None:
        if not NETWORKX_AVAILABLE:
            raise RuntimeError("networkx is required but not installed")
        _memory_instance = GraphMemory()
    return _memory_instance


async def demo():
    """Demo usage of graph memory"""
    print("=== Graph Memory Demo ===\n")
    
    gm = get_graph_memory()
    
    base_time = time.time_ns()
    
    # Store some historical memories
    memories_data = [
        {
            'regime': 'high_vol_bull',
            'vol': 0.8, 'trend': 0.9, 'liq': 0.7,
            'pnl': 0.05, 'actions': ['reduce_size', 'widen_spreads'],
            'lessons': ['Volatility spikes require smaller positions']
        },
        {
            'regime': 'high_vol_bull',
            'vol': 0.85, 'trend': 0.8, 'liq': 0.6,
            'pnl': -0.03, 'actions': ['hold_position'],
            'lessons': ['Holding through volatility can be costly']
        },
        {
            'regime': 'flash_crash',
            'vol': 2.0, 'trend': -0.9, 'liq': 0.2,
            'pnl': -0.15, 'actions': ['emergency_close'],
            'lessons': ['Fast execution critical in flash crashes']
        },
        {
            'regime': 'low_vol_consolidation',
            'vol': 0.2, 'trend': 0.1, 'liq': 0.9,
            'pnl': 0.01, 'actions': ['accumulate'],
            'lessons': ['Patient accumulation works in calm markets']
        },
    ]
    
    for i, data in enumerate(memories_data):
        state = MarketState(
            timestamp_ns=base_time - i * 86400 * 1e9,
            volatility=data['vol'],
            trend=data['trend'],
            liquidity_score=data['liq'],
            regime_type=data['regime'],
            recent_events=[]
        )
        
        memory = HistoricalMemory(
            memory_id=f"mem_{i}",
            state=state,
            outcome={'pnl': data['pnl'], 'slippage_bps': 10},
            actions_taken=data['actions'],
            lessons_learned=data['lessons'],
            importance=0.8,
            last_access_ns=base_time - i * 86400 * 1e9
        )
        
        await gm.store_memory(memory)
    
    # Get stats
    stats = gm.get_memory_stats()
    print(f"Memory Stats:")
    print(f"  Stored: {stats['num_memories']} memories")
    print(f"  Edges: {stats['num_edges']}")
    print(f"  Capacity: {stats['capacity_used_pct']:.1f}%")
    
    # Query with current state
    current = MarketState(
        timestamp_ns=base_time,
        volatility=0.75,
        trend=0.85,
        liquidity_score=0.65,
        regime_type='high_vol_bull',
        recent_events=['fed_speech']
    )
    
    context = await gm.retrieve_context(current)
    
    print(f"\nRetrieved Context:")
    print(f"  Confidence: {context.confidence_score:.2f}")
    print(f"  Similar Memories: {len(context.similar_memories)}")
    print(f"  Recommendations: {context.recommended_actions}")
    print(f"  Warnings: {context.risk_warnings}")
    print(f"  Retrieval Time: {context.retrieval_time_ms:.1f}ms")


if __name__ == "__main__":
    asyncio.run(demo())
