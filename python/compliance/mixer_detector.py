"""
Mixer Detector - Heuristic graph analyzer for detecting Tornado Cash and mixer interactions.
Blocks toxic DeFi settlements by analyzing transaction patterns and counterparty relationships.
Memory-efficient design targeting <100MB RAM footprint.
"""
import numpy as np
import logging
from typing import Dict, List, Set, Optional, Tuple, Any
from pathlib import Path
from collections import defaultdict, deque
import time

logger = logging.getLogger(__name__)


class TransactionGraph:
    """
    Lightweight transaction graph for tracking address relationships.
    Uses adjacency lists for memory efficiency.
    """
    
    def __init__(self, max_nodes: int = 100_000):
        self.max_nodes = max_nodes
        
        # Adjacency list representation
        self._adjacency: Dict[str, Set[str]] = defaultdict(set)
        self._reverse_adjacency: Dict[str, Set[str]] = defaultdict(set)
        
        # Node metadata
        self._node_flags: Dict[str, int] = {}
        self._node_scores: Dict[str, float] = {}
        
        # Known mixer addresses
        self._mixer_addresses: Set[str] = set()
        
        self._node_count = 0
        
        # Flags
        self.FLAG_MIXER = 1
        self.FLAG_HIGH_RISK = 2
        self.FLAG_SANCTIONED = 4
    
    def add_transaction(self, from_addr: str, to_addr: str, 
                        amount: float = 0.0, timestamp: float = None) -> None:
        """Add a transaction edge to the graph."""
        if self._node_count >= self.max_nodes:
            self._prune_old_nodes()
        
        # Add nodes if new
        if from_addr not in self._node_flags:
            self._node_flags[from_addr] = 0
            self._node_scores[from_addr] = 0.0
            self._node_count += 1
        
        if to_addr not in self._node_flags:
            self._node_flags[to_addr] = 0
            self._node_scores[to_addr] = 0.0
            self._node_count += 1
        
        # Add edge
        self._adjacency[from_addr].add(to_addr)
        self._reverse_adjacency[to_addr].add(from_addr)
    
    def mark_as_mixer(self, address: str) -> None:
        """Mark an address as a known mixer."""
        self._mixer_addresses.add(address.lower())
        self._node_flags[address.lower()] |= self.FLAG_MIXER
        self._node_scores[address.lower()] = 1.0
    
    def is_mixer(self, address: str) -> bool:
        """Check if address is marked as mixer."""
        return address.lower() in self._mixer_addresses
    
    def get_neighbors(self, address: str, direction: str = 'out') -> Set[str]:
        """Get neighboring addresses."""
        addr_lower = address.lower()
        if direction == 'out':
            return self._adjacency.get(addr_lower, set())
        elif direction == 'in':
            return self._reverse_adjacency.get(addr_lower, set())
        else:
            return self._adjacency.get(addr_lower, set()) | self._reverse_adjacency.get(addr_lower, set())
    
    def calculate_risk_score(self, address: str, depth: int = 3) -> float:
        """
        Calculate mixer interaction risk score using BFS.
        
        Args:
            address: Address to analyze
            depth: How many hops to trace
            
        Returns:
            Risk score from 0.0 (clean) to 1.0 (high risk)
        """
        addr_lower = address.lower()
        
        # Check if directly flagged
        if self._node_flags.get(addr_lower, 0) & self.FLAG_MIXER:
            return 1.0
        
        # BFS to find mixer connections
        visited = set()
        queue = deque([(addr_lower, 0)])  # (address, distance)
        
        mixer_connections = []
        
        while queue:
            current, dist = queue.popleft()
            
            if current in visited or dist > depth:
                continue
            
            visited.add(current)
            
            # Check if this is a mixer
            if current in self._mixer_addresses:
                mixer_connections.append((current, dist))
                continue
            
            # Explore neighbors with decay factor
            if dist < depth:
                for neighbor in self._adjacency.get(current, set()):
                    if neighbor not in visited:
                        queue.append((neighbor, dist + 1))
                
                for neighbor in self._reverse_adjacency.get(current, set()):
                    if neighbor not in visited:
                        queue.append((neighbor, dist + 1))
        
        # Calculate risk score based on proximity to mixers
        if not mixer_connections:
            return 0.0
        
        # Exponential decay based on distance
        risk = sum(0.5 ** dist for _, dist in mixer_connections)
        risk = min(1.0, risk / len(mixer_connections))  # Normalize
        
        return risk
    
    def _prune_old_nodes(self) -> None:
        """Remove low-activity nodes to stay within memory limits."""
        # Simple pruning: remove nodes with no flags and low connectivity
        to_remove = []
        
        for addr in list(self._node_flags.keys())[:self.max_nodes // 10]:
            if (self._node_flags[addr] == 0 and 
                len(self._adjacency.get(addr, set())) < 2 and
                len(self._reverse_adjacency.get(addr, set())) < 2):
                to_remove.append(addr)
        
        for addr in to_remove:
            # Remove from adjacency lists
            for neighbor in self._adjacency.get(addr, set()):
                self._reverse_adjacency[neighbor].discard(addr)
            for neighbor in self._reverse_adjacency.get(addr, set()):
                self._adjacency[neighbor].discard(addr)
            
            del self._adjacency[addr]
            del self._reverse_adjacency[addr]
            del self._node_flags[addr]
            del self._node_scores[addr]
            self._node_count -= 1


class MixerDetector:
    """
    Detects interactions with Tornado Cash and other mixers.
    Uses heuristic graph analysis to identify toxic DeFi settlement patterns.
    """
    
    # Known mixer addresses (sample - would be populated from intelligence)
    TORNADO_CASH_ROUTERS = {
        '0xd90e2f925da726b50c4ed8d0fb90ad053324f31b',  # TC Router
        '0x47ce0c6ed5b0ce3d3a51fdb1c52dc66a7c3c2936',  # TC ETH
        '0x12d66f87a04a9e220743712ce6d9bb1b5616b8fc',  # TC ETH 0.1
        '0x47ba1ded40098d91294aeeb76e8b9719c5bc04bd',  # TC ETH 1
        '0xfd8610db23984d75b8e9171f6b4f3ea73d8c351f',  # TC ETH 10
        '0x375777f36927f2525a95130286e28d769ae6b6d2',  # TC ETH 100
    }
    
    OTHER_MIXERS = {
        # Blender.io
        '0x6222c4b68d25a08f29adb110935ab52684cba5e7',
        # Sinbad.io
        '0x3889927f095EB0C332fB1Ad6926Acf4D1d25E38c',
        # Railgun
        '0x1b9d5f7f1e3c7f5e5f5f5f5f5f5f5f5f5f5f5f5f',
    }
    
    def __init__(self, graph_max_nodes: int = 100_000,
                 risk_threshold: float = 0.5,
                 auto_populate: bool = True):
        """
        Initialize mixer detector.
        
        Args:
            graph_max_nodes: Maximum nodes in transaction graph
            risk_threshold: Threshold for blocking decisions
            auto_populate: Whether to populate with known mixers
        """
        self.risk_threshold = risk_threshold
        
        # Initialize transaction graph
        self.graph = TransactionGraph(max_nodes=graph_max_nodes)
        
        # Statistics
        self._checks_count = 0
        self._blocks_count = 0
        self._high_risk_count = 0
        
        # Populate known mixers
        if auto_populate:
            self._populate_mixer_addresses()
        
        logger.info(f"MixerDetector initialized with threshold={risk_threshold}")
    
    def _populate_mixer_addresses(self) -> None:
        """Populate graph with known mixer addresses."""
        for addr in self.TORNADO_CASH_ROUTERS:
            self.graph.mark_as_mixer(addr)
        
        for addr in self.OTHER_MIXERS:
            self.graph.mark_as_mixer(addr)
        
        logger.info(f"Populated {len(self.TORNADO_CASH_ROUTERS) + len(self.OTHER_MIXERS)} mixer addresses")
    
    def add_transaction(self, from_addr: str, to_addr: str, 
                        amount: float = 0.0, timestamp: float = None) -> None:
        """Record a transaction for graph analysis."""
        self.graph.add_transaction(from_addr, to_addr, amount, timestamp)
    
    def analyze(self, address: str, 
                transaction_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Analyze an address for mixer interactions.
        
        Args:
            address: Address to analyze
            transaction_history: Optional list of transactions to add to graph
            
        Returns:
            Analysis results dictionary
        """
        self._checks_count += 1
        
        addr_lower = address.lower()
        
        # Add transaction history to graph if provided
        if transaction_history:
            for tx in transaction_history:
                self.add_transaction(
                    tx.get('from', ''),
                    tx.get('to', ''),
                    tx.get('amount', 0.0),
                    tx.get('timestamp')
                )
        
        # Calculate risk score
        risk_score = self.graph.calculate_risk_score(address, depth=3)
        
        # Check direct mixer interaction
        is_direct_mixer = self.graph.is_mixer(address)
        
        # Analyze transaction patterns
        pattern_analysis = self._analyze_patterns(address)
        
        # Determine action
        is_high_risk = risk_score >= self.risk_threshold or is_direct_mixer
        action = 'BLOCK' if is_high_risk else 'ALLOW'
        
        if is_high_risk:
            self._blocks_count += 1
            if risk_score >= 0.8:
                self._high_risk_count += 1
        
        result = {
            'address': address,
            'risk_score': float(risk_score),
            'is_direct_mixer': is_direct_mixer,
            'is_high_risk': is_high_risk,
            'action': action,
            'pattern_analysis': pattern_analysis,
            'reason': self._get_block_reason(risk_score, is_direct_mixer, pattern_analysis)
        }
        
        return result
    
    def _analyze_patterns(self, address: str) -> Dict[str, Any]:
        """Analyze transaction patterns for mixer-like behavior."""
        addr_lower = address.lower()
        
        # Get transaction counts
        out_degree = len(self.graph._adjacency.get(addr_lower, set()))
        in_degree = len(self.graph._reverse_adjacency.get(addr_lower, set()))
        
        # Pattern indicators
        patterns = {
            'high_out_degree': out_degree > 50,
            'high_in_degree': in_degree > 50,
            'balanced_flow': 0.8 < (out_degree / max(1, in_degree)) < 1.2 if in_degree > 0 else False,
            'rapid_turnover': False  # Would need timestamp data
        }
        
        # Mixer-like patterns
        is_mixer_like = (
            patterns['high_out_degree'] and 
            patterns['high_in_degree'] and 
            patterns['balanced_flow']
        )
        
        return {
            'out_degree': out_degree,
            'in_degree': in_degree,
            'patterns': patterns,
            'is_mixer_like': is_mixer_like
        }
    
    def _get_block_reason(self, risk_score: float, is_direct_mixer: bool,
                          pattern_analysis: Dict) -> Optional[str]:
        """Generate human-readable block reason."""
        reasons = []
        
        if is_direct_mixer:
            reasons.append("Direct mixer address")
        
        if risk_score >= 0.8:
            reasons.append(f"High mixer proximity (score: {risk_score:.2f})")
        elif risk_score >= self.risk_threshold:
            reasons.append(f"Mixer interaction detected (score: {risk_score:.2f})")
        
        if pattern_analysis.get('is_mixer_like'):
            reasons.append("Mixer-like transaction patterns")
        
        return "; ".join(reasons) if reasons else None
    
    def check_settlement(self, from_addr: str, to_addr: str,
                         amount: float = 0.0) -> Dict[str, Any]:
        """
        Check if a settlement between two addresses should be blocked.
        
        Args:
            from_addr: Sender address
            to_addr: Receiver address
            amount: Transaction amount
            
        Returns:
            Settlement check results
        """
        # Analyze both parties
        from_result = self.analyze(from_addr)
        to_result = self.analyze(to_addr)
        
        # Determine if settlement should be blocked
        should_block = (
            from_result['is_high_risk'] or 
            to_result['is_high_risk']
        )
        
        if should_block:
            self._blocks_count += 1
        
        return {
            'from_address': from_addr,
            'to_address': to_addr,
            'amount': amount,
            'should_block': should_block,
            'from_risk': from_result,
            'to_risk': to_result,
            'block_reason': from_result['reason'] or to_result['reason']
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            'total_checks': self._checks_count,
            'addresses_blocked': self._blocks_count,
            'high_risk_addresses': self._high_risk_count,
            'block_rate': self._blocks_count / max(1, self._checks_count),
            'graph_nodes': self.graph._node_count,
            'known_mixers': len(self.graph._mixer_addresses)
        }
    
    def add_known_mixer(self, address: str, source: str = 'manual') -> None:
        """Add a known mixer address."""
        self.graph.mark_as_mixer(address)
        logger.info(f"Added known mixer: {address} (source: {source})")
    
    def export_graph(self, path: str) -> None:
        """Export transaction graph to file."""
        import pickle
        
        data = {
            'adjacency': dict(self.graph._adjacency),
            'reverse_adjacency': dict(self.graph._reverse_adjacency),
            'node_flags': self.graph._node_flags,
            'node_scores': self.graph._node_scores,
            'mixer_addresses': self.graph._mixer_addresses
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Exported graph to {path}")
    
    def import_graph(self, path: str) -> None:
        """Import transaction graph from file."""
        import pickle
        
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            self.graph._adjacency = defaultdict(set, data.get('adjacency', {}))
            self.graph._reverse_adjacency = defaultdict(set, data.get('reverse_adjacency', {}))
            self.graph._node_flags = data.get('node_flags', {})
            self.graph._node_scores = data.get('node_scores', {})
            self.graph._mixer_addresses = data.get('mixer_addresses', set())
            self.graph._node_count = len(self.graph._node_flags)
            
            logger.info(f"Imported graph from {path}")
        except Exception as e:
            logger.error(f"Failed to import graph: {e}")


# Singleton instance
_mixer_detector: Optional[MixerDetector] = None


def get_mixer_detector(config: Optional[Dict] = None) -> MixerDetector:
    """Get or create singleton MixerDetector instance."""
    global _mixer_detector
    if _mixer_detector is None:
        config = config or {}
        _mixer_detector = MixerDetector(
            graph_max_nodes=config.get('graph_max_nodes', 100_000),
            risk_threshold=config.get('risk_threshold', 0.5),
            auto_populate=config.get('auto_populate', True)
        )
    return _mixer_detector


def reset_mixer_detector() -> None:
    """Reset singleton instance."""
    global _mixer_detector
    _mixer_detector = None


__all__ = ['MixerDetector', 'TransactionGraph', 'get_mixer_detector', 'reset_mixer_detector']
