"""
SOUL.md Mistake Penalty Integration
Integrates parsed SOUL.md mistake logs to apply massive negative rewards when the 
RL agent repeats historical toxic behaviors.

Forces PPO execution agents to actively avoid order book shapes and regimes that 
previously resulted in adverse selection or slippage.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import deque
import hashlib
import json


@dataclass
class MistakeEntry:
    """Represents a single mistake entry from SOUL.md."""
    timestamp: float
    regime_hash: str  # Hash of market regime features
    orderbook_shape: str  # Encoded order book state
    action_taken: int
    consequence: str  # e.g., "adverse_selection", "slippage", "toxic_flow"
    severity: float  # 0.0 to 1.0
    pnl_impact: float
    
    def matches(self, 
                current_regime_hash: str,
                current_orderbook_hash: str,
                current_action: int,
                tolerance: float = 0.8) -> bool:
        """Check if current state matches this mistake."""
        # Exact action match required
        if self.action_taken != current_action:
            return False
        
        # Regime similarity check
        regime_match = self._hash_similarity(self.regime_hash, current_regime_hash)
        
        # Order book shape similarity
        ob_match = self._hash_similarity(self.orderbook_shape, current_orderbook_hash)
        
        # Both must exceed tolerance
        return regime_match >= tolerance and ob_match >= tolerance
    
    @staticmethod
    def _hash_similarity(hash1: str, hash2: str) -> float:
        """Compute similarity between two hex hashes (0-1)."""
        if hash1 == hash2:
            return 1.0
        
        # Compare prefix bytes
        min_len = min(len(hash1), len(hash2))
        matching_bytes = sum(1 for i in range(min_len) if hash1[i] == hash2[i])
        return matching_bytes / min_len


@dataclass
class SOULConfig:
    """Configuration for SOUL penalty system."""
    # Penalty multipliers by consequence type
    consequence_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'adverse_selection': 5.0,
        'slippage': 3.0,
        'toxic_flow': 4.0,
        'inventory_buildup': 2.0,
        'missed_opportunity': 1.0
    })
    
    # Severity thresholds
    min_severity_threshold: float = 0.5  # Only penalize for severe mistakes
    
    # Matching parameters
    regime_tolerance: float = 0.7
    orderbook_tolerance: float = 0.6
    
    # Memory management
    max_mistakes_stored: int = 10000
    decay_half_life_steps: int = 500
    
    # Look-ahead prevention
    min_age_steps: int = 5  # Mistakes must be at least N steps old


class SOULMistakeDatabase:
    """
    Database of historical mistakes from SOUL.md parsing.
    Efficiently stores and retrieves mistake patterns.
    """
    
    def __init__(self, config: Optional[SOULConfig] = None):
        self.config = config or SOULConfig()
        
        # Mistake storage with bounded size
        self._mistakes: deque = deque(maxlen=self.config.max_mistakes_stored)
        self._mistake_index: Dict[str, List[int]] = {}  # Index by regime hash prefix
        
        # Step counter for age tracking
        self._current_step = 0
        
        # Statistics
        self._match_count = 0
        self._total_penalty_applied = 0.0
        
    def add_mistake(self, mistake: MistakeEntry):
        """Add a new mistake to the database."""
        idx = len(self._mistakes)
        self._mistakes.append(mistake)
        
        # Index by regime hash prefix (first 4 chars)
        prefix = mistake.regime_hash[:4]
        if prefix not in self._mistake_index:
            self._mistake_index[prefix] = []
        self._mistake_index[prefix].append(idx)
        
    def load_from_soul_log(self, soul_log_path: str):
        """
        Load mistakes from parsed SOUL.md log file.
        
        Expected format (JSON lines):
        {"timestamp": ..., "regime_features": [...], "orderbook_state": "...", ...}
        """
        try:
            with open(soul_log_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    entry = json.loads(line.strip())
                    
                    # Compute regime hash from features
                    regime_features = entry.get('regime_features', [])
                    regime_hash = self._compute_feature_hash(regime_features)
                    
                    mistake = MistakeEntry(
                        timestamp=entry.get('timestamp', 0),
                        regime_hash=regime_hash,
                        orderbook_shape=entry.get('orderbook_state', ''),
                        action_taken=entry.get('action', 0),
                        consequence=entry.get('consequence', 'unknown'),
                        severity=entry.get('severity', 0.5),
                        pnl_impact=entry.get('pnl_impact', 0.0)
                    )
                    
                    if mistake.severity >= self.config.min_severity_threshold:
                        self.add_mistake(mistake)
                        
        except FileNotFoundError:
            pass  # No log file yet, start empty
            
    def _compute_feature_hash(self, features: List[float]) -> str:
        """Compute deterministic hash from feature vector."""
        # Quantize features to reduce sensitivity
        quantized = [round(f * 100) for f in features]
        data = json.dumps(quantized, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()
    
    def find_matching_mistakes(self,
                               current_regime_hash: str,
                               current_orderbook_hash: str,
                               current_action: int) -> List[MistakeEntry]:
        """Find all matching historical mistakes."""
        matches = []
        
        # Search indexed entries first
        prefix = current_regime_hash[:4]
        candidate_indices = self._mistake_index.get(prefix, [])
        
        for idx in candidate_indices:
            if idx < len(self._mistakes):
                mistake = self._mistakes[idx]
                
                # Check age constraint (prevent look-ahead bias)
                age = self._current_step - idx
                if age < self.config.min_age_steps:
                    continue
                
                if mistake.matches(
                    current_regime_hash,
                    current_orderbook_hash,
                    current_action,
                    self.config.regime_tolerance
                ):
                    matches.append(mistake)
        
        # Also do a linear scan of recent mistakes not in index
        recent_start = max(0, len(self._mistakes) - 1000)
        for idx in range(recent_start, len(self._mistakes)):
            if idx in candidate_indices:
                continue  # Already checked
                
            mistake = self._mistakes[idx]
            age = self._current_step - idx
            
            if age < self.config.min_age_steps:
                continue
            
            if mistake.matches(
                current_regime_hash,
                current_orderbook_hash,
                current_action,
                self.config.regime_tolerance
            ):
                matches.append(mistake)
        
        return matches
    
    def step(self):
        """Advance step counter."""
        self._current_step += 1
        
    def get_statistics(self) -> Dict[str, any]:
        """Get database statistics."""
        return {
            'total_mistakes': len(self._mistakes),
            'match_count': self._match_count,
            'total_penalty_applied': self._total_penalty_applied,
            'current_step': self._current_step
        }


class SOULPenaltyEngine:
    """
    Computes penalties based on SOUL.md mistake matching.
    Integrates with RL reward pipeline.
    """
    
    def __init__(self, config: Optional[SOULConfig] = None):
        self.config = config or SOULConfig()
        self.database = SOULMistakeDatabase(config)
        
        # Feature hasher for real-time computation
        self._feature_dim = 0
        
        # Decay factor for older mistakes
        self._decay_factor = np.exp(-np.log(2) / self.config.decay_half_life_steps)
        
    def compute_penalty(self,
                        regime_features: np.ndarray,
                        orderbook_features: np.ndarray,
                        action: int) -> Tuple[float, Dict]:
        """
        Compute SOUL penalty for current state-action pair.
        
        Args:
            regime_features: Current market regime features
            orderbook_features: Current order book state features
            action: Action being considered
            
        Returns:
            Penalty value (negative) and metadata
        """
        # Compute hashes
        regime_hash = self.database._compute_feature_hash(regime_features.tolist())
        orderbook_hash = self.database._compute_feature_hash(orderbook_features.tolist())
        
        # Find matching mistakes
        matches = self.database.find_matching_mistakes(
            regime_hash,
            orderbook_hash,
            action
        )
        
        if not matches:
            return 0.0, {'matches_found': 0}
        
        # Compute weighted penalty
        total_penalty = 0.0
        penalty_breakdown = []
        
        for mistake in matches:
            # Base penalty from severity
            base_penalty = mistake.severity * mistake.pnl_impact
            
            # Apply consequence multiplier
            multiplier = self.config.consequence_multipliers.get(
                mistake.consequence, 1.0
            )
            
            # Apply time decay
            age = self.database._current_step - len(self.database._mistakes) + \
                  list(self.database._mistakes).index(mistake)
            decay = self._decay_factor ** age
            
            # Final penalty
            step_penalty = base_penalty * multiplier * decay
            total_penalty += step_penalty
            
            penalty_breakdown.append({
                'consequence': mistake.consequence,
                'severity': mistake.severity,
                'multiplier': multiplier,
                'decay': decay,
                'penalty': step_penalty
            })
        
        # Update statistics
        self.database._match_count += len(matches)
        self.database._total_penalty_applied += total_penalty
        
        # Return as negative value (penalty)
        metadata = {
            'matches_found': len(matches),
            'total_penalty': -total_penalty,
            'breakdown': penalty_breakdown[:5]  # Top 5 matches
        }
        
        return -total_penalty, metadata
    
    def encode_regime(self,
                      volatility: float,
                      momentum: float,
                      volume_ratio: float,
                      spread_bps: float) -> np.ndarray:
        """Encode regime metrics into feature vector."""
        features = np.array([
            volatility,
            momentum,
            volume_ratio,
            spread_bps,
            volatility * momentum,  # Interaction term
            volume_ratio / (spread_bps + 1e-6)  # Liquidity term
        ], dtype=np.float32)
        
        self._feature_dim = len(features)
        return features
    
    def encode_orderbook(self,
                         bid_imbalance: float,
                         ask_imbalance: float,
                         mid_spread: float,
                         depth_ratio: float) -> np.ndarray:
        """Encode order book state into feature vector."""
        features = np.array([
            bid_imbalance,
            ask_imbalance,
            mid_spread,
            depth_ratio,
            bid_imbalance - ask_imbalance,  # Net imbalance
            (bid_imbalance + ask_imbalance) / 2  # Total pressure
        ], dtype=np.float32)
        
        return features
    
    def record_mistake(self,
                       regime_features: np.ndarray,
                       orderbook_features: np.ndarray,
                       action: int,
                       consequence: str,
                       severity: float,
                       pnl_impact: float):
        """Record a new mistake after it occurs."""
        regime_hash = self.database._compute_feature_hash(regime_features.tolist())
        orderbook_hash = self.database._compute_feature_hash(orderbook_features.tolist())
        
        mistake = MistakeEntry(
            timestamp=self.database._current_step,
            regime_hash=regime_hash,
            orderbook_shape=orderbook_hash,
            action_taken=action,
            consequence=consequence,
            severity=severity,
            pnl_impact=pnl_impact
        )
        
        if severity >= self.config.min_severity_threshold:
            self.database.add_mistake(mistake)
    
    def step(self):
        """Advance internal step counter."""
        self.database.step()
    
    def reset(self):
        """Reset penalty engine state."""
        self.database = SOULMistakeDatabase(self.config)


# Module exports
__all__ = [
    'MistakeEntry',
    'SOULConfig',
    'SOULMistakeDatabase',
    'SOULPenaltyEngine'
]
