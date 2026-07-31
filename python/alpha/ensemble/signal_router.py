"""
Dynamic Signal Router based on HMM Regime State.
Weights StatArb, Lead-Lag, and Vol signals based on active market regime.
Implements regime-adaptive portfolio allocation for alpha signals.
Memory-efficient with bounded signal queues.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class MarketRegime(Enum):
    """Hidden Markov Model regime states."""
    BULL_LOW_VOL = "bull_low_vol"      # Risk-on, calm
    BULL_HIGH_VOL = "bull_high_vol"    # Risk-on, volatile
    BEAR_LOW_VOL = "bear_low_vol"      # Risk-off, calm
    BEAR_HIGH_VOL = "bear_high_vol"    # Risk-off, volatile (panic)
    TRANSITION = "transition"          # Regime change in progress
    UNKNOWN = "unknown"                # Insufficient data


class SignalCategory(Enum):
    """Categories of alpha signals."""
    STATARB = "statarb"           # Mean reversion / pairs
    LEADLAG = "leadlag"           # Cross-asset momentum
    VOLATILITY = "volatility"     # Vol-based directional
    ONCHAIN = "onchain"           # On-chain structural
    MOMENTUM = "momentum"         # Trend following


@dataclass
class RegimeWeights:
    """Signal weights for a specific regime."""
    statarb_weight: float
    leadlag_weight: float
    volatility_weight: float
    onchain_weight: float
    momentum_weight: float
    
    def normalize(self) -> 'RegimeWeights':
        """Normalize weights to sum to 1."""
        total = (self.statarb_weight + self.leaddag_weight + 
                self.volatility_weight + self.onchain_weight + 
                self.momentum_weight)
        if total > 0:
            return RegimeWeights(
                statarb_weight=self.statarb_weight / total,
                leadlag_weight=self.leadlag_weight / total,
                volatility_weight=self.volatility_weight / total,
                onchain_weight=self.onchain_weight / total,
                momentum_weight=self.momentum_weight / total
            )
        return self


# Default regime weights (calibrated from historical analysis)
REGIME_WEIGHTS = {
    MarketRegime.BULL_LOW_VOL: RegimeWeights(0.15, 0.25, 0.20, 0.20, 0.20),
    MarketRegime.BULL_HIGH_VOL: RegimeWeights(0.30, 0.15, 0.25, 0.15, 0.15),
    MarketRegime.BEAR_LOW_VOL: RegimeWeights(0.20, 0.20, 0.15, 0.25, 0.20),
    MarketRegime.BEAR_HIGH_VOL: RegimeWeights(0.35, 0.10, 0.20, 0.20, 0.15),
    MarketRegime.TRANSITION: RegimeWeights(0.20, 0.20, 0.20, 0.20, 0.20),
    MarketRegime.UNKNOWN: RegimeWeights(0.20, 0.20, 0.20, 0.20, 0.20),
}


@dataclass
class RoutedSignal:
    """Signal after regime-based routing."""
    original_signal: Dict
    category: SignalCategory
    regime: MarketRegime
    applied_weight: float
    adjusted_confidence: float
    final_score: float
    should_execute: bool
    execution_priority: int  # 1=highest


class HiddenMarkovModel:
    """
    Simple HMM for regime detection.
    Uses observable market features to infer hidden regime state.
    """
    
    def __init__(self, 
                 n_states: int = 5,
                 transition_memory: int = 50):
        """
        Args:
            n_states: Number of hidden states (regimes)
            transition_memory: Samples for transition probability estimation
        """
        self.n_states = n_states
        
        # State tracking
        self.current_state = MarketRegime.UNKNOWN
        self.state_history = []
        self.max_history = 500
        
        # Transition matrix (simplified - would be learned)
        self.transition_matrix = self._default_transition_matrix()
        
        # Emission parameters (mean/var for each feature per state)
        self.emission_params = self._default_emission_params()
        
        # Recent observations for inference
        self.observation_buffer = []
        self.buffer_size = transition_memory
    
    def _default_transition_matrix(self) -> np.ndarray:
        """Default transition probabilities between regimes."""
        # High diagonal = sticky regimes
        return np.array([
            [0.85, 0.10, 0.03, 0.02, 0.00],  # Bull low vol
            [0.15, 0.70, 0.05, 0.10, 0.00],  # Bull high vol
            [0.05, 0.05, 0.80, 0.10, 0.00],  # Bear low vol
            [0.02, 0.15, 0.15, 0.68, 0.00],  # Bear high vol
            [0.20, 0.20, 0.20, 0.20, 0.20],  # Transition
        ])
    
    def _default_emission_params(self) -> Dict:
        """Default emission distribution parameters."""
        # For each regime: (mean_return, mean_vol, mean_correlation)
        return {
            MarketRegime.BULL_LOW_VOL: {'ret': 0.001, 'vol': 0.3, 'corr': 0.5},
            MarketRegime.BULL_HIGH_VOL: {'ret': 0.002, 'vol': 0.8, 'corr': 0.7},
            MarketRegime.BEAR_LOW_VOL: {'ret': -0.001, 'vol': 0.4, 'corr': 0.6},
            MarketRegime.BEAR_HIGH_VOL: {'ret': -0.003, 'vol': 1.2, 'corr': 0.9},
            MarketRegime.TRANSITION: {'ret': 0.0, 'vol': 0.5, 'corr': 0.5},
        }
    
    def update_and_infer(self,
                         market_return: float,
                         volatility: float,
                         correlation: float) -> MarketRegime:
        """
        Update HMM with new observation and infer current regime.
        
        Args:
            market_return: Recent market return
            volatility: Current volatility level
            correlation: Cross-asset correlation
            
        Returns:
            Inferred MarketRegime
        """
        observation = np.array([market_return, volatility, correlation])
        self.observation_buffer.append(observation)
        
        if len(self.observation_buffer) > self.buffer_size:
            self.observation_buffer.pop(0)
        
        if len(self.observation_buffer) < 10:
            return MarketRegime.UNKNOWN
        
        # Calculate likelihood of observation under each regime
        log_likelihoods = {}
        
        for regime, params in self.emission_params.items():
            # Simplified Gaussian likelihood
            diff_ret = market_return - params['ret']
            diff_vol = volatility - params['vol']
            diff_corr = correlation - params['corr']
            
            # Negative squared distance (log likelihood approximation)
            ll = -(diff_ret**2 / 0.001 + diff_vol**2 / 0.5 + diff_corr**2 / 0.3)
            log_likelihoods[regime] = ll
        
        # Find most likely regime
        best_regime = max(log_likelihoods.keys(), key=lambda k: log_likelihoods[k])
        
        # Apply transition smoothing
        if self.state_history:
            prev_state = self.state_history[-1]
            if prev_state != best_regime:
                # Check if transition is significant enough
                ll_diff = log_likelihoods[best_regime] - log_likelihoods[prev_state]
                if ll_diff < 0.5:  # Not confident enough to switch
                    best_regime = prev_state
        
        # Update history
        self.state_history.append(best_regime)
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)
        
        self.current_state = best_regime
        return best_regime
    
    def get_regime_probability(self) -> Dict[MarketRegime, float]:
        """Get probability distribution over regimes."""
        if not self.observation_buffer:
            return {r: 0.2 for r in MarketRegime}
        
        # Calculate normalized likelihoods
        recent = np.mean(self.observation_buffer[-10:], axis=0)
        market_return, volatility, correlation = recent
        
        log_likelihoods = {}
        for regime, params in self.emission_params.items():
            diff_ret = market_return - params['ret']
            diff_vol = volatility - params['vol']
            diff_corr = correlation - params['corr']
            ll = np.exp(-(diff_ret**2 / 0.001 + diff_vol**2 / 0.5 + diff_corr**2 / 0.3))
            log_likelihoods[regime] = ll
        
        # Normalize to probabilities
        total = sum(log_likelihoods.values())
        if total > 0:
            return {k: v/total for k, v in log_likelihoods.items()}
        return {r: 0.2 for r in MarketRegime}


class SignalRouter:
    """
    Routes and weights alpha signals based on HMM regime.
    Implements dynamic allocation across signal categories.
    """
    
    def __init__(self, min_confidence_threshold: float = 0.3):
        """
        Args:
            min_confidence_threshold: Minimum confidence to consider signal
        """
        self.min_confidence_threshold = min_confidence_threshold
        
        # Initialize HMM
        self.hmm = HiddenMarkovModel()
        
        # Current regime weights
        self.current_weights = REGIME_WEIGHTS[MarketRegime.UNKNOWN]
        
        # Signal queues by category
        self.signal_queues = {cat: [] for cat in SignalCategory}
        self.max_queue_size = 100
        
        # Routing statistics
        self.stats = {
            'signals_routed': 0,
            'signals_filtered': 0,
            'by_category': {cat.value: 0 for cat in SignalCategory}
        }
    
    def update_regime(self,
                      market_return: float,
                      volatility: float,
                      correlation: float) -> MarketRegime:
        """Update regime inference with latest market data."""
        regime = self.hmm.update_and_infer(market_return, volatility, correlation)
        self.current_weights = REGIME_WEIGHTS[regime]
        return regime
    
    def route_signal(self,
                     signal: Dict,
                     category: SignalCategory,
                     timestamp_ns: int) -> Optional[RoutedSignal]:
        """
        Route a signal through regime-based weighting.
        
        Args:
            signal: Primary signal dictionary
            category: Signal category
            timestamp_ns: Signal timestamp
            
        Returns:
            RoutedSignal or None if filtered
        """
        # Check confidence threshold
        confidence = signal.get('confidence', 0.0)
        if confidence < self.min_confidence_threshold:
            self.stats['signals_filtered'] += 1
            return None
        
        # Get weight for this category
        weight_map = {
            SignalCategory.STATARb: self.current_weights.statarb_weight,
            SignalCategory.LEADLAG: self.current_weights.leadlag_weight,
            SignalCategory.VOLATILITY: self.current_weights.volatility_weight,
            SignalCategory.ONCHAIN: self.current_weights.onchain_weight,
            SignalCategory.MOMENTUM: self.current_weights.momentum_weight,
        }
        
        applied_weight = weight_map.get(category, 0.2)
        
        # Calculate adjusted confidence
        adjusted_confidence = confidence * (1 + applied_weight)
        adjusted_confidence = min(adjusted_confidence, 1.0)
        
        # Calculate final score (confidence * weight * signal_strength)
        strength = signal.get('strength', 0.5)
        final_score = adjusted_confidence * strength * applied_weight * 2
        
        # Determine execution priority
        if final_score > 0.7:
            priority = 1
            should_execute = True
        elif final_score > 0.4:
            priority = 2
            should_execute = True
        elif final_score > 0.2:
            priority = 3
            should_execute = True
        else:
            priority = 4
            should_execute = False
        
        routed = RoutedSignal(
            original_signal=signal,
            category=category,
            regime=self.hmm.current_state,
            applied_weight=applied_weight,
            adjusted_confidence=adjusted_confidence,
            final_score=final_score,
            should_execute=should_execute,
            execution_priority=priority
        )
        
        # Add to queue
        queue = self.signal_queues[category]
        queue.append(routed)
        if len(queue) > self.max_queue_size:
            queue.pop(0)
        
        # Update stats
        self.stats['signals_routed'] += 1
        self.stats['by_category'][category.value] += 1
        
        return routed
    
    def get_top_signals(self, n: int = 10) -> List[RoutedSignal]:
        """Get top N signals by final score across all categories."""
        all_signals = []
        for queue in self.signal_queues.values():
            all_signals.extend(queue)
        
        # Sort by final score descending
        all_signals.sort(key=lambda s: s.final_score, reverse=True)
        
        return all_signals[:n]
    
    def get_signals_by_category(self, category: SignalCategory) -> List[RoutedSignal]:
        """Get all signals for a specific category."""
        return list(self.signal_queues[category])
    
    def get_execution_queue(self) -> List[RoutedSignal]:
        """Get ordered execution queue (only executable signals)."""
        executable = []
        for queue in self.signal_queues.values():
            executable.extend([s for s in queue if s.should_execute])
        
        # Sort by priority, then by score
        executable.sort(key=lambda s: (s.execution_priority, -s.final_score))
        
        return executable
    
    def get_statistics(self) -> Dict:
        """Get router statistics."""
        return {
            **self.stats,
            'current_regime': self.hmm.current_state.value,
            'regime_weights': {
                'statarb': self.current_weights.statarb_weight,
                'leadlag': self.current_weights.leadlag_weight,
                'volatility': self.current_weights.volatility_weight,
                'onchain': self.current_weights.onchain_weight,
                'momentum': self.current_weights.momentum_weight,
            },
            'queue_sizes': {cat.value: len(q) for cat, q in self.signal_queues.items()},
            'regime_probs': self.hmm.get_regime_probability()
        }


__all__ = [
    'SignalRouter',
    'HiddenMarkovModel',
    'MarketRegime',
    'SignalCategory',
    'RoutedSignal',
    'RegimeWeights',
    'REGIME_WEIGHTS'
]
