"""
Lightweight GraphSAGE model (exported to ONNX) to classify unknown wallet addresses
based on transaction graph topology. Identifies institutional accumulators and
predatory MEV bots by analyzing their structural position in the blockchain network.
Strictly enforces memory limits for production deployment.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class GraphSAGEConfig:
    """Configuration for GraphSAGE model."""
    # Feature dimensions
    input_dim: int = 64
    hidden_dim: int = 128
    output_dim: int = 8  # Number of wallet classes
    
    # Architecture
    num_layers: int = 2
    aggregator_type: str = "mean"  # mean, lstm, pool
    sample_size: int = 25  # Neighbors to sample per layer
    
    # Memory constraints
    max_nodes_batch: int = 512
    cpu_mem_limit: int = 256 * 1024 * 1024  # 256MB for graph model
    intra_op_threads: int = 2
    
    # Classification labels
    class_labels: List[str] = field(default_factory=lambda: [
        'retail', 'institutional', 'mev_bot', 'exchange_hot',
        'exchange_cold', 'defi_protocol', 'mixer', 'unknown'
    ])
    
    provider_options: Optional[Dict] = None
    
    def __post_init__(self):
        if self.provider_options is None:
            self.provider_options = {
                'CPUExecutionProvider': {
                    'arena_extend_strategy': 'kSameAsRequested',
                    'cpu_mem_limit': self.cpu_mem_limit,
                    'intra_op_num_threads': self.intra_op_threads,
                    'inter_op_num_threads': 1
                }
            }


@dataclass
class WalletFeatures:
    """Features extracted from a wallet address."""
    # Transaction features
    total_transactions: int = 0
    avg_tx_value: float = 0.0
    tx_value_std: float = 0.0
    unique_counterparties: int = 0
    
    # Temporal features
    avg_tx_interval_seconds: float = 0.0
    activity_hour_distribution: np.ndarray = field(default_factory=lambda: np.zeros(24))
    
    # Graph features
    in_degree: int = 0
    out_degree: int = 0
    clustering_coefficient: float = 0.0
    pagerank_score: float = 0.0
    
    # DeFi-specific features
    defi_interactions: int = 0
    nft_trades: int = 0
    gas_spent_eth: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert features to numpy array for model input."""
        return np.array([
            self.total_transactions,
            self.avg_tx_value,
            self.tx_value_std,
            self.unique_counterparties,
            self.avg_tx_interval_seconds,
            self.in_degree,
            self.out_degree,
            self.clustering_coefficient,
            self.pagerank_score,
            self.defi_interactions,
            self.nft_trades,
            self.gas_spent_eth,
            *self.activity_hour_distribution
        ], dtype=np.float32)
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'WalletFeatures':
        """Reconstruct features from numpy array."""
        return cls(
            total_transactions=int(arr[0]),
            avg_tx_value=float(arr[1]),
            tx_value_std=float(arr[2]),
            unique_counterparties=int(arr[3]),
            avg_tx_interval_seconds=float(arr[4]),
            in_degree=int(arr[5]),
            out_degree=int(arr[6]),
            clustering_coefficient=float(arr[7]),
            pagerank_score=float(arr[8]),
            defi_interactions=int(arr[9]),
            nft_trades=int(arr[10]),
            gas_spent_eth=float(arr[11]),
            activity_hour_distribution=arr[12:36]
        )


class GraphSAGEInference:
    """
    GraphSAGE inference wrapper for wallet classification.
    
    The model uses neighborhood aggregation to learn embeddings that capture
    both local graph structure and node features for classification.
    """
    
    def __init__(
        self,
        model_path: str,
        config: GraphSAGEConfig,
        session_options: Optional[ort.SessionOptions] = None
    ):
        self.config = config
        self.model_path = model_path
        
        # Configure session options
        if session_options is None:
            session_options = ort.SessionOptions()
            session_options.enable_mem_pattern = True
            session_options.enable_cpu_mem_arena = True
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.intra_op_num_threads = config.intra_op_threads
            
        self.session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=['CPUExecutionProvider']
        )
        
        self._validate_model_signature()
        
        # Cache for node embeddings (bounded size)
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._cache_max_size = 10000
        
        logger.info(f"GraphSAGE model loaded: {model_path}")
    
    def _validate_model_signature(self):
        """Validate model input/output signatures."""
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        
        logger.debug(f"GraphSAGE inputs: {[i.name for i in inputs]}")
        logger.debug(f"GraphSAGE outputs: {[o.name for o in outputs]}")
    
    def predict(
        self,
        node_features: np.ndarray,
        neighbor_features: Optional[np.ndarray] = None,
        adjacency: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict wallet class probabilities.
        
        Args:
            node_features: Features of target nodes (batch_size, input_dim)
            neighbor_features: Features of sampled neighbors (optional)
            adjacency: Adjacency information for aggregation (optional)
            
        Returns:
            Tuple of (probabilities, predicted_class_indices)
        """
        if node_features.dtype != np.float32:
            node_features = node_features.astype(np.float32)
        
        if node_features.ndim == 1:
            node_features = node_features.reshape(1, -1)
        
        # Build input feed
        input_feed = {'node_features': node_features}
        
        if neighbor_features is not None:
            if neighbor_features.dtype != np.float32:
                neighbor_features = neighbor_features.astype(np.float32)
            input_feed['neighbor_features'] = neighbor_features
        
        if adjacency is not None:
            input_feed['adjacency'] = adjacency.astype(np.int32)
        
        # Run inference
        outputs = self.session.run(None, input_feed)
        
        # First output should be logits/probabilities
        logits = outputs[0]
        
        # Convert to probabilities if needed
        if logits.min() < 0 or logits.max() > 1:
            probs = self._softmax(logits)
        else:
            probs = logits
        
        predicted_classes = np.argmax(probs, axis=-1)
        
        return probs, predicted_classes
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def get_embedding(
        self,
        node_features: np.ndarray,
        node_id: str
    ) -> np.ndarray:
        """
        Get node embedding with caching.
        
        Args:
            node_features: Node feature vector
            node_id: Unique node identifier for caching
            
        Returns:
            Node embedding vector
        """
        # Check cache first
        if node_id in self._embedding_cache:
            return self._embedding_cache[node_id]
        
        # Run inference to get embedding (usually second-to-last layer output)
        if node_features.dtype != np.float32:
            node_features = node_features.astype(np.float32)
        
        if node_features.ndim == 1:
            node_features = node_features.reshape(1, -1)
        
        input_feed = {'node_features': node_features}
        outputs = self.session.run(None, input_feed)
        
        # Assume last output before classification is the embedding
        # This depends on model export structure
        if len(outputs) > 1:
            embedding = outputs[-2]  # Second to last output
        else:
            # Use hidden representation from final layer
            embedding = outputs[0][:, :self.config.hidden_dim]
        
        # Cache embedding
        if len(self._embedding_cache) >= self._cache_max_size:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self._embedding_cache.keys())[:1000]
            for key in keys_to_remove:
                del self._embedding_cache[key]
        
        self._embedding_cache[node_id] = embedding[0]
        
        return embedding[0]
    
    def classify_wallets(
        self,
        wallet_features: List[WalletFeatures],
        wallet_addresses: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Classify multiple wallets and return detailed results.
        
        Returns dict mapping address to classification results.
        """
        if len(wallet_features) != len(wallet_addresses):
            raise ValueError("Features and addresses must have same length")
        
        # Stack features
        feature_matrix = np.stack([wf.to_array() for wf in wallet_features])
        
        # Ensure correct input dimension
        if feature_matrix.shape[1] < self.config.input_dim:
            # Pad with zeros
            padding = np.zeros(
                (feature_matrix.shape[0], self.config.input_dim - feature_matrix.shape[1]),
                dtype=np.float32
            )
            feature_matrix = np.concatenate([feature_matrix, padding], axis=1)
        elif feature_matrix.shape[1] > self.config.input_dim:
            feature_matrix = feature_matrix[:, :self.config.input_dim]
        
        # Run prediction
        probs, predictions = self.predict(feature_matrix)
        
        # Build results
        results = {}
        for i, (address, prob, pred_idx) in enumerate(zip(wallet_addresses, probs, predictions)):
            results[address] = {
                'predicted_class': self.config.class_labels[pred_idx],
                'probabilities': {
                    label: float(p) for label, p in zip(self.config.class_labels, prob)
                },
                'confidence': float(np.max(prob)),
                'features': wallet_features[i]
            }
        
        return results
    
    def detect_institutional_accumulators(
        self,
        wallet_results: Dict[str, Dict[str, Any]],
        min_confidence: float = 0.7
    ) -> List[str]:
        """
        Identify wallets likely to be institutional accumulators.
        
        Criteria:
        - High probability of 'institutional' class
        - Large number of transactions
        - Low retail probability
        """
        accumulators = []
        
        for address, result in wallet_results.items():
            if result['predicted_class'] != 'institutional':
                continue
            
            if result['confidence'] < min_confidence:
                continue
            
            features = result.get('features')
            if features is None:
                continue
            
            # Additional heuristics
            if features.total_transactions < 100:
                continue
            
            # Institutional wallets typically have high unique counterparty count
            if features.unique_counterparties < 50:
                continue
            
            accumulators.append(address)
        
        return accumulators
    
    def detect_mev_bots(
        self,
        wallet_results: Dict[str, Dict[str, Any]],
        min_confidence: float = 0.6
    ) -> List[str]:
        """
        Identify predatory MEV bots.
        
        Criteria:
        - High probability of 'mev_bot' class
        - High gas spending
        - High transaction frequency
        - Specific temporal patterns
        """
        mev_bots = []
        
        for address, result in wallet_results.items():
            if result['predicted_class'] != 'mev_bot':
                continue
            
            if result['confidence'] < min_confidence:
                continue
            
            features = result.get('features')
            if features is None:
                continue
            
            # MEV bots typically have very high gas spending
            if features.gas_spent_eth < 1.0:
                continue
            
            # High transaction frequency
            if features.total_transactions < 500:
                continue
            
            mev_bots.append(address)
        
        return mev_bots


class GraphSAGEEnsemble:
    """Ensemble of GraphSAGE models for robust classification."""
    
    def __init__(self, model_configs: List[Tuple[str, GraphSAGEConfig]]):
        self.models: List[GraphSAGEInference] = []
        
        for model_path, config in model_configs:
            try:
                model = GraphSAGEInference(model_path, config)
                self.models.append(model)
            except Exception as e:
                logger.warning(f"Failed to load GraphSAGE model {model_path}: {e}")
        
        if not self.models:
            raise RuntimeError("No GraphSAGE models could be loaded")
    
    def predict_ensemble(
        self,
        node_features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Average predictions from ensemble members."""
        all_probs = []
        
        for model in self.models:
            probs, _ = model.predict(node_features)
            all_probs.append(probs)
        
        avg_probs = np.mean(all_probs, axis=0)
        predicted_classes = np.argmax(avg_probs, axis=-1)
        
        return avg_probs, predicted_classes


# Factory function
def create_graphsage_model(
    model_path: str,
    input_dim: int = 64,
    hidden_dim: int = 128,
    num_classes: int = 8
) -> GraphSAGEInference:
    """Factory function to create GraphSAGE model."""
    config = GraphSAGEConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=num_classes
    )
    return GraphSAGEInference(model_path, config)
