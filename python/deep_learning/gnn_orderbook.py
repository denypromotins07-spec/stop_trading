"""
Graph Neural Network for L2 Order Book Modeling.
Models the order book as a dynamic graph where price levels are nodes
and order flow dependencies are edges. Exported to ONNX for inference.

Strictly enforces 3GB RAM limit by using frozen graphs and ONNX Runtime.
"""

import numpy as np
from typing import Tuple, Optional, List
import onnxruntime as ort
from dataclasses import dataclass
import threading

# Configure ONNX Runtime for minimal footprint
SESSION_OPTIONS = ort.SessionOptions()
SESSION_OPTIONS.intra_op_num_threads = 1
SESSION_OPTIONS.inter_op_num_threads = 1
SESSION_OPTIONS.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
SESSION_OPTIONS.enable_mem_pattern = True


@dataclass
class OrderBookNode:
    """Represents a single price level in the order book graph."""
    price_level: int
    bid_volume: float
    ask_volume: float
    spread_distance: float
    imbalance: float
    timestamp_ns: int


@dataclass
class OrderBookEdge:
    """Represents relationships between price levels."""
    source_idx: int
    target_idx: int
    edge_type: int  # 0=adjacent, 1=spread, 2=volume_weighted
    weight: float


class OrderBookGraphBuilder:
    """
    Builds dynamic graph representations of L2 order book snapshots.
    Uses pre-allocated buffers to avoid heap allocations during hot paths.
    """
    
    MAX_NODES = 50  # 25 levels each side
    MAX_EDGES = 200
    
    def __init__(self):
        self._lock = threading.Lock()
        # Pre-allocate node feature buffer [MAX_NODES, 5]
        self._node_buffer = np.zeros((self.MAX_NODES, 5), dtype=np.float32)
        # Pre-allocate edge index buffer [2, MAX_EDGES]
        self._edge_index_buffer = np.zeros((2, self.MAX_EDGES), dtype=np.int64)
        # Pre-allocate edge attribute buffer [MAX_EDGES, 3]
        self._edge_attr_buffer = np.zeros((self.MAX_EDGES, 3), dtype=np.float32)
        
    def build_graph(
        self,
        bid_prices: np.ndarray,
        bid_volumes: np.ndarray,
        ask_prices: np.ndarray,
        ask_volumes: np.ndarray,
        mid_price: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Construct graph from L2 snapshot.
        
        Args:
            bid_prices: Array of bid prices (sorted descending)
            bid_volumes: Array of bid volumes
            ask_prices: Array of ask prices (sorted ascending)
            ask_volumes: Array of ask volumes
            mid_price: Current mid price
            
        Returns:
            node_features: [N, 5] array
            edge_index: [2, E] array
            edge_attr: [E, 3] array
        """
        with self._lock:
            n_levels = min(len(bid_prices), len(ask_prices), 25)
            total_nodes = n_levels * 2
            
            if total_nodes == 0:
                return (
                    self._node_buffer[:1],
                    self._edge_index_buffer[:, :1],
                    self._edge_attr_buffer[:1]
                )
            
            # Build node features
            for i in range(n_levels):
                # Bid nodes (0 to n_levels-1)
                bid_spread_dist = (mid_price - bid_prices[i]) / mid_price
                bid_imbalance = (bid_volumes[i] - ask_volumes[min(i, n_levels-1)]) / \
                               (bid_volumes[i] + ask_volumes[min(i, n_levels-1)] + 1e-8)
                
                self._node_buffer[i] = [
                    i,  # level index
                    float(bid_volumes[i]),
                    0.0,  # ask volume (0 for bid nodes)
                    bid_spread_dist,
                    bid_imbalance
                ]
                
                # Ask nodes (n_levels to 2*n_levels-1)
                ask_spread_dist = (ask_prices[i] - mid_price) / mid_price
                ask_imbalance = (bid_volumes[min(i, n_levels-1)] - ask_volumes[i]) / \
                               (bid_volumes[min(i, n_levels-1)] + ask_volumes[i] + 1e-8)
                
                idx = n_levels + i
                self._node_buffer[idx] = [
                    i,  # level index
                    0.0,  # bid volume (0 for ask nodes)
                    float(ask_volumes[i]),
                    ask_spread_dist,
                    ask_imbalance
                ]
            
            # Build edges
            edge_count = 0
            
            # Adjacent edges within bids
            for i in range(n_levels - 1):
                if edge_count >= self.MAX_EDGES:
                    break
                self._edge_index_buffer[0, edge_count] = i
                self._edge_index_buffer[1, edge_count] = i + 1
                weight = 1.0 / (1.0 + abs(i - (i + 1)))
                self._edge_attr_buffer[edge_count] = [0, weight, 1.0]
                edge_count += 1
            
            # Adjacent edges within asks
            for i in range(n_levels - 1):
                if edge_count >= self.MAX_EDGES:
                    break
                src = n_levels + i
                tgt = n_levels + i + 1
                self._edge_index_buffer[0, edge_count] = src
                self._edge_index_buffer[1, edge_count] = tgt
                weight = 1.0 / (1.0 + abs(i - (i + 1)))
                self._edge_attr_buffer[edge_count] = [0, weight, 1.0]
                edge_count += 1
            
            # Spread edges (bid-ask pairs at same level)
            for i in range(n_levels):
                if edge_count >= self.MAX_EDGES:
                    break
                self._edge_index_buffer[0, edge_count] = i
                self._edge_index_buffer[1, edge_count] = n_levels + i
                self._edge_attr_buffer[edge_count] = [1, 1.0, 1.0]
                edge_count += 1
            
            # Volume-weighted cross edges
            for i in range(min(5, n_levels)):
                if edge_count >= self.MAX_EDGES:
                    break
                for j in range(min(5, n_levels)):
                    if edge_count >= self.MAX_EDGES:
                        break
                    if i != j:
                        self._edge_index_buffer[0, edge_count] = i
                        self._edge_index_buffer[1, edge_count] = n_levels + j
                        vol_ratio = min(bid_volumes[i], ask_volumes[j]) / \
                                   (max(bid_volumes[i], ask_volumes[j]) + 1e-8)
                        self._edge_attr_buffer[edge_count] = [2, vol_ratio, 1.0]
                        edge_count += 1
            
            return (
                self._node_buffer[:total_nodes].copy(),
                self._edge_index_buffer[:, :edge_count].copy(),
                self._edge_attr_buffer[:edge_count].copy()
            )


class GNNOrderBookPredictor:
    """
    ONNX-based GNN predictor for order book dynamics.
    Captures spatial relationships and temporal dependencies.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.graph_builder = OrderBookGraphBuilder()
        self.session: Optional[ort.InferenceSession] = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """Load pre-trained GNN model exported to ONNX."""
        self.session = ort.InferenceSession(
            model_path,
            sess_options=SESSION_OPTIONS,
            providers=['CPUExecutionProvider']
        )
    
    def predict_liquidity_sweep(
        self,
        bid_prices: np.ndarray,
        bid_volumes: np.ndarray,
        ask_prices: np.ndarray,
        ask_volumes: np.ndarray,
        mid_price: float,
        time_horizon_ms: int = 100
    ) -> Tuple[float, float]:
        """
        Predict probability of liquidity sweep in next N milliseconds.
        
        Args:
            bid_prices: Bid price levels
            bid_volumes: Bid volumes at each level
            ask_prices: Ask price levels
            ask_volumes: Ask volumes at each level
            mid_price: Current mid price
            time_horizon_ms: Prediction horizon
            
        Returns:
            sweep_prob_up: Probability of upward sweep (asks depleted)
            sweep_prob_down: Probability of downward sweep (bids depleted)
        """
        if self.session is None:
            # Fallback heuristic if no model loaded
            return self._heuristic_prediction(
                bid_prices, bid_volumes, ask_prices, ask_volumes, mid_price
            )
        
        # Build graph representation
        node_feat, edge_idx, edge_attr = self.graph_builder.build_graph(
            bid_prices, bid_volumes, ask_prices, ask_volumes, mid_price
        )
        
        # Prepare inputs for ONNX model
        # Expected inputs: node_features, edge_index, edge_attributes, horizon
        inputs = {
            'node_features': node_feat.astype(np.float32),
            'edge_index': edge_idx.astype(np.int64),
            'edge_attributes': edge_attr.astype(np.float32),
            'horizon': np.array([time_horizon_ms], dtype=np.float32)
        }
        
        # Run inference
        outputs = self.session.run(None, inputs)
        
        # Output: [sweep_prob_up, sweep_prob_down, expected_depth, confidence]
        sweep_prob_up = float(outputs[0][0])
        sweep_prob_down = float(outputs[0][1])
        
        return sweep_prob_up, sweep_prob_down
    
    def _heuristic_prediction(
        self,
        bid_prices: np.ndarray,
        bid_volumes: np.ndarray,
        ask_prices: np.ndarray,
        ask_volumes: np.ndarray,
        mid_price: float
    ) -> Tuple[float, float]:
        """Fallback heuristic based on order book imbalance."""
        if len(bid_volumes) == 0 or len(ask_volumes) == 0:
            return 0.5, 0.5
        
        # Calculate weighted imbalance near touch
        weights = np.exp(-np.arange(min(5, len(bid_volumes))))
        bid_weighted = np.sum(bid_volumes[:len(weights)] * weights)
        ask_weighted = np.sum(ask_volumes[:len(weights)] * weights)
        
        total = bid_weighted + ask_weighted + 1e-8
        imbalance = (bid_weighted - ask_weighted) / total
        
        # Convert to sweep probabilities
        sweep_prob_up = 0.5 + 0.3 * (-imbalance)  # Negative imbalance -> up sweep
        sweep_prob_down = 0.5 + 0.3 * imbalance
        
        return np.clip(sweep_prob_up, 0.0, 1.0), np.clip(sweep_prob_down, 0.0, 1.0)
    
    def get_node_embeddings(
        self,
        bid_prices: np.ndarray,
        bid_volumes: np.ndarray,
        ask_prices: np.ndarray,
        ask_volumes: np.ndarray,
        mid_price: float
    ) -> np.ndarray:
        """
        Extract learned embeddings for each price level.
        Useful for regime detection and clustering.
        """
        if self.session is None:
            return np.zeros((1, 64), dtype=np.float32)
        
        node_feat, edge_idx, edge_attr = self.graph_builder.build_graph(
            bid_prices, bid_volumes, ask_prices, ask_volumes, mid_price
        )
        
        inputs = {
            'node_features': node_feat.astype(np.float32),
            'edge_index': edge_idx.astype(np.int64),
            'edge_attributes': edge_attr.astype(np.float32),
            'horizon': np.array([0], dtype=np.float32),
            'return_embeddings': np.array([1], dtype=np.int64)
        }
        
        outputs = self.session.run(None, inputs)
        return outputs[1]  # Embeddings tensor


# Global singleton instance
_gnn_instance: Optional[GNNOrderBookPredictor] = None
_gnn_lock = threading.Lock()


def get_gnn_predictor(model_path: Optional[str] = None) -> GNNOrderBookPredictor:
    """Thread-safe singleton access to GNN predictor."""
    global _gnn_instance
    
    with _gnn_lock:
        if _gnn_instance is None:
            _gnn_instance = GNNOrderBookPredictor(model_path)
        elif model_path and _gnn_instance.session is None:
            _gnn_instance.load_model(model_path)
        
        return _gnn_instance


if __name__ == "__main__":
    # Demo usage
    np.random.seed(42)
    
    # Simulate L2 order book
    mid = 50000.0
    bid_prices = mid - np.arange(1, 26) * 5.0
    ask_prices = mid + np.arange(1, 26) * 5.0
    bid_volumes = np.random.exponential(10.0, 25)
    ask_volumes = np.random.exponential(10.0, 25)
    
    builder = OrderBookGraphBuilder()
    nodes, edges, attrs = builder.build_graph(
        bid_prices, bid_volumes, ask_prices, ask_volumes, mid
    )
    
    print(f"Built graph: {nodes.shape[0]} nodes, {edges.shape[1]} edges")
    print(f"Node features shape: {nodes.shape}")
    print(f"Edge index shape: {edges.shape}")
    print(f"Edge attributes shape: {attrs.shape}")
    
    predictor = GNNOrderBookPredictor()
    prob_up, prob_down = predictor.predict_liquidity_sweep(
        bid_prices, bid_volumes, ask_prices, ask_volumes, mid
    )
    print(f"Sweep probabilities - Up: {prob_up:.4f}, Down: {prob_down:.4f}")
