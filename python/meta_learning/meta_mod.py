"""
Chapter 3: Meta-Learning & Few-Shot Adaptation (MAML)
File: python/meta_learning/meta_mod.py

Module root for meta-learning infrastructure.
Manages the meta-weight registry and triggers fast-adaptation routines
when HMM regime detector flags a novel state.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import numpy as np

# Import local modules
from .maml_trainer import MAMLConfig, MAMLTrainer, MAMLModel
from .task_distribution import (
    MarketRegime,
    TaskDistributionConfig,
    TaskSampler,
    TaskDefinition
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MetaWeightEntry:
    """Registry entry for meta-learned weights."""
    weight_id: str
    regime: MarketRegime
    symbol: str
    created_at: datetime
    base_weights: Dict[str, np.ndarray]
    adaptation_count: int = 0
    last_adapted: Optional[datetime] = None
    performance_score: float = 0.0


@dataclass
class RegimeDetectionResult:
    """Result from HMM regime detection."""
    current_regime: MarketRegime
    confidence: float
    is_novel: bool  # True if regime differs significantly from training
    novelty_score: float  # 0=familiar, 1=completely novel
    recommended_action: str  # "hold", "adapt", "retrain"


class MetaWeightRegistry:
    """
    Registry for storing and retrieving meta-learned weights.
    Supports fast lookup by regime and symbol.
    """
    
    def __init__(self, max_entries_per_regime: int = 10):
        self.max_entries = max_entries_per_regime
        self.weights: Dict[str, MetaWeightEntry] = {}
        self.regime_index: Dict[MarketRegime, List[str]] = {
            r: [] for r in MarketRegime
        }
        self.symbol_index: Dict[str, List[str]] = {}
    
    def register(
        self,
        weight_id: str,
        regime: MarketRegime,
        symbol: str,
        base_weights: Dict[str, np.ndarray]
    ) -> MetaWeightEntry:
        """Register new meta weights."""
        entry = MetaWeightEntry(
            weight_id=weight_id,
            regime=regime,
            symbol=symbol,
            created_at=datetime.utcnow(),
            base_weights=base_weights
        )
        
        self.weights[weight_id] = entry
        
        # Update indices
        if len(self.regime_index[regime]) >= self.max_entries:
            # Remove oldest
            old_id = self.regime_index[regime].pop(0)
            if old_id in self.weights:
                del self.weights[old_id]
        
        self.regime_index[regime].append(weight_id)
        
        if symbol not in self.symbol_index:
            self.symbol_index[symbol] = []
        self.symbol_index[symbol].append(weight_id)
        
        logger.info(f"Registered meta weights: {weight_id}")
        return entry
    
    def get_by_regime(
        self, 
        regime: MarketRegime
    ) -> List[MetaWeightEntry]:
        """Get all weights for a regime."""
        ids = self.regime_index.get(regime, [])
        return [self.weights[id] for id in ids if id in self.weights]
    
    def get_best_for_regime(
        self,
        regime: MarketRegime
    ) -> Optional[MetaWeightEntry]:
        """Get best performing weights for a regime."""
        entries = self.get_by_regime(regime)
        if not entries:
            return None
        return max(entries, key=lambda e: e.performance_score)
    
    def update_performance(
        self,
        weight_id: str,
        score: float
    ):
        """Update performance score for weights."""
        if weight_id in self.weights:
            self.weights[weight_id].performance_score = score
            self.weights[weight_id].last_adapted = datetime.utcnow()
    
    def increment_adaptation_count(self, weight_id: str):
        """Increment adaptation count."""
        if weight_id in self.weights:
            self.weights[weight_id].adaptation_count += 1


class MetaLearningSystem:
    """
    Main meta-learning system orchestrator.
    Manages MAML training, fast adaptation, and weight registry.
    """
    
    def __init__(
        self,
        maml_config: Optional[MAMLConfig] = None,
        task_config: Optional[TaskDistributionConfig] = None
    ):
        # Initialize components
        self.maml_trainer = MAMLTrainer(maml_config)
        self.task_sampler = TaskSampler(task_config)
        self.weight_registry = MetaWeightRegistry()
        
        # State
        self.is_initialized = False
        self.current_regime: Optional[MarketRegime] = None
        self.adapted_model: Optional[MAMLModel] = None
        
        # Callbacks
        self.hmm_detector: Optional[Callable] = None
        self.on_adaptation_complete: Optional[Callable] = None
    
    def initialize(self):
        """Initialize the meta-learning system."""
        self.is_initialized = True
        logger.info("Meta-learning system initialized")
    
    def detect_regime(
        self,
        market_features: np.ndarray
    ) -> RegimeDetectionResult:
        """
        Detect current market regime using HMM or similar.
        
        Args:
            market_features: Current market feature vector
        
        Returns:
            RegimeDetectionResult with regime and novelty assessment
        """
        if self.hmm_detector:
            result = self.hmm_detector(market_features)
            return RegimeDetectionResult(**result)
        
        # Default: assume sideways market
        return RegimeDetectionResult(
            current_regime=MarketRegime.SIDEWAYS,
            confidence=0.5,
            is_novel=False,
            novelty_score=0.0,
            recommended_action="hold"
        )
    
    async def adapt_to_regime(
        self,
        regime: MarketRegime,
        support_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        n_steps: int = 5
    ) -> MAMLModel:
        """
        Fast adaptation to detected regime.
        
        Args:
            regime: Detected market regime
            support_data: Optional few-shot data for adaptation
            n_steps: Number of gradient steps
        
        Returns:
            Adapted model
        """
        logger.info(f"Adapting to regime: {regime.value}")
        
        # Check registry for existing weights
        best_entry = self.weight_registry.get_best_for_regime(regime)
        
        if best_entry and support_data is None:
            # Use existing weights without further adaptation
            logger.info(f"Using cached weights: {best_entry.weight_id}")
            self.maml_trainer.model.load_params(
                {k: torch.FloatTensor(v) for k, v in best_entry.base_weights.items()}
            )
            self.adapted_model = self.maml_trainer.model
        elif support_data is not None:
            # Perform few-shot adaptation
            self.adapted_model = self.maml_trainer.adapt_to_task(
                support_data,
                n_steps=n_steps
            )
            
            # Register new weights
            weight_id = f"{regime.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            self.weight_registry.register(
                weight_id=weight_id,
                regime=regime,
                symbol="multi",
                base_weights=self.maml_trainer.get_base_weights()
            )
            self.weight_registry.increment_adaptation_count(weight_id)
        else:
            # No data available, use base model
            logger.warning("No support data, using base model")
            self.adapted_model = self.maml_trainer.model
        
        # Notify completion
        if self.on_adaptation_complete:
            await self.on_adaptation_complete(regime, self.adapted_model)
        
        return self.adapted_model
    
    async def handle_novel_regime(
        self,
        regime_result: RegimeDetectionResult,
        market_features: np.ndarray
    ):
        """
        Handle detection of novel regime requiring rapid adaptation.
        """
        if not regime_result.is_novel:
            return
        
        logger.warning(
            f"Novel regime detected! Novelty score: {regime_result.novelty_score:.2f}"
        )
        
        if regime_result.recommended_action == "retrain":
            # Trigger background retraining
            logger.info("Initiating background meta-training...")
            asyncio.create_task(self._background_meta_training())
        
        elif regime_result.recommended_action == "adapt":
            # Generate synthetic support data from current features
            # In production, would use recent historical data
            synthetic_x = np.tile(market_features, (10, 1))
            synthetic_y = np.zeros(10, dtype=np.int64)  # Placeholder labels
            
            await self.adapt_to_regime(
                regime_result.current_regime,
                support_data=(synthetic_x, synthetic_y),
                n_steps=10  # More steps for novel regimes
            )
    
    async def _background_meta_training(self):
        """Run meta-training in background."""
        try:
            # Sample diverse task batch
            tasks = self.task_sampler.sample_task_batch(batch_size=4)
            if not tasks:
                logger.warning("No tasks available for meta-training")
                return
            
            # Prepare batch
            batch = self.task_sampler.prepare_maml_batch(tasks)
            
            # Perform meta-update
            if batch:
                loss = self.maml_trainer.meta_update(batch)
                logger.info(f"Background meta-training complete, loss: {loss:.4f}")
                
                # Register updated weights
                for task in tasks:
                    weight_id = f"meta_{task.regime.value}_{datetime.utcnow().strftime('%H%M%S')}"
                    self.weight_registry.register(
                        weight_id=weight_id,
                        regime=task.regime,
                        symbol=task.symbol,
                        base_weights=self.maml_trainer.get_base_weights()
                    )
        
        except Exception as e:
            logger.error(f"Background meta-training failed: {e}")
    
    def predict(
        self,
        features: np.ndarray,
        use_adapted: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make prediction using current model.
        
        Args:
            features: Input features
            use_adapted: Use adapted model if available
        
        Returns:
            (predictions, probabilities)
        """
        model = self.adapted_model if use_adapted and self.adapted_model else self.maml_trainer.model
        
        if model is None:
            raise RuntimeError("No model available for prediction")
        
        return self.maml_trainer.predict(model, features)
    
    def save_checkpoint(self, path: str):
        """Save full meta-learning checkpoint."""
        self.maml_trainer.save_checkpoint(path)
        
        # Also save registry state
        registry_state = {
            entry.weight_id: {
                "regime": entry.regime.value,
                "symbol": entry.symbol,
                "created_at": entry.created_at.isoformat(),
                "adaptation_count": entry.adaptation_count,
                "performance_score": entry.performance_score
            }
            for entry in self.weight_registry.weights.values()
        }
        
        import json
        with open(path + ".registry.json", "w") as f:
            json.dump(registry_state, f, indent=2)
        
        logger.info(f"Meta-learning checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Load meta-learning checkpoint."""
        self.maml_trainer.load_checkpoint(path)
        logger.info(f"Meta-learning checkpoint loaded: {path}")


# Export for module use
__all__ = [
    "MetaWeightEntry",
    "RegimeDetectionResult",
    "MetaWeightRegistry",
    "MetaLearningSystem"
]
