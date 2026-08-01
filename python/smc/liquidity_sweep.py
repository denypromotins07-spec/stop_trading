"""
Liquidity Sweep Detector - Hidden Markov Model for classifying liquidity sweeps and stop-hunts.
Filters out fake retail breakouts from genuine institutional liquidity grabs.
Memory-efficient implementation targeting <50MB RAM footprint.
"""
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import hmmlearn for HMM, fall back to custom implementation
try:
    from hmmlearn import hmm
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False
    logger.info("hmmlearn not available, using custom HMM implementation")


class CustomHMM:
    """
    Lightweight Hidden Markov Model implementation for liquidity sweep detection.
    Avoids heavy dependencies while maintaining accuracy.
    """
    
    def __init__(self, n_states: int = 3, n_iter: int = 100):
        self.n_states = n_states
        self.n_iter = n_iter
        
        # Initialize model parameters
        self.transmat_ = None  # Transition matrix
        self.means_ = None     # Emission means
        self.covars_ = None    # Emission covariances
        self.startprob_ = None # Initial state probabilities
        
        self._trained = False
    
    def fit(self, X: np.ndarray) -> 'CustomHMM':
        """
        Fit HMM using Baum-Welch algorithm.
        
        Args:
            X: Observation sequences (n_samples, n_features)
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters randomly
        rng = np.random.RandomState(42)
        self.startprob_ = np.ones(self.n_states) / self.n_states
        self.transmat_ = np.eye(self.n_states) * 0.7 + (1 - np.eye(self.n_states)) * 0.15
        self.means_ = rng.randn(self.n_states, n_features)
        self.covars_ = np.ones((self.n_states, n_features)) + 0.1
        
        # Simple Baum-Welch iterations
        for _ in range(self.n_iter):
            # E-step: Compute posteriors
            log_prob, posteriors = self._e_step(X)
            
            # M-step: Update parameters
            self._m_step(X, posteriors)
        
        self._trained = True
        return self
    
    def _e_step(self, X: np.ndarray) -> Tuple[float, np.ndarray]:
        """E-step: Compute state posteriors using forward-backward."""
        n_samples = len(X)
        
        # Forward pass (simplified)
        alpha = np.zeros((n_samples, self.n_states))
        alpha[0] = self.startprob_ * self._emission_prob(X[0])
        
        for t in range(1, n_samples):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t-1] * self.transmat_[:, j]) * self._emission_prob(X[t])[j]
        
        # Normalize
        log_prob = np.log(np.sum(alpha[-1]) + 1e-10)
        
        # Backward pass and posterior computation (simplified)
        posteriors = alpha / (np.sum(alpha, axis=1, keepdims=True) + 1e-10)
        
        return log_prob, posteriors
    
    def _emission_prob(self, x: np.ndarray) -> np.ndarray:
        """Compute emission probabilities (Gaussian)."""
        probs = np.zeros(self.n_states)
        for k in range(self.n_states):
            diff = x - self.means_[k]
            inv_cov = 1.0 / (self.covars_[k] + 1e-10)
            mahalanobis = np.sum(diff ** 2 * inv_cov)
            probs[k] = np.exp(-0.5 * mahalanobis) / (np.sqrt(2 * np.pi * self.covars_[k]) + 1e-10)
        return probs
    
    def _m_step(self, X: np.ndarray, posteriors: np.ndarray) -> None:
        """M-step: Update model parameters."""
        n_samples, n_features = X.shape
        
        # Update initial probabilities
        self.startprob_ = posteriors[0]
        
        # Update transition matrix
        for i in range(self.n_states):
            for j in range(self.n_states):
                # Simplified transition update
                self.transmat_[i, j] = np.mean(posteriors[:, j])
        
        # Normalize transitions
        self.transmat_ /= np.sum(self.transmat_, axis=1, keepdims=True) + 1e-10
        
        # Update emission parameters
        for k in range(self.n_states):
            resp = posteriors[:, k:k+1]
            total_resp = np.sum(resp) + 1e-10
            
            self.means_[k] = np.sum(resp * X, axis=0) / total_resp
            self.covars_[k] = np.sum(resp * (X - self.means_[k]) ** 2, axis=0) / total_resp
            self.covars_[k] = np.maximum(self.covars_[k], 0.01)  # Prevent collapse
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict most likely state sequence."""
        if not self._trained:
            raise ValueError("Model not trained")
        
        n_samples = len(X)
        states = np.zeros(n_samples, dtype=int)
        
        # Viterbi algorithm (simplified)
        delta = np.zeros((n_samples, self.n_states))
        delta[0] = np.log(self.startprob_ + 1e-10) + np.log(self._emission_prob(X[0]) + 1e-10)
        
        for t in range(1, n_samples):
            for j in range(self.n_states):
                delta[t, j] = np.max(delta[t-1] + np.log(self.transmat_[:, j] + 1e-10)) + \
                             np.log(self._emission_prob(X[t])[j] + 1e-10)
        
        # Backtrack
        states[-1] = np.argmax(delta[-1])
        for t in range(n_samples - 2, -1, -1):
            states[t] = np.argmax(delta[t+1] + np.log(self.transmat_[:, states[t+1]] + 1e-10))
        
        return states
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict state probabilities."""
        if not self._trained:
            raise ValueError("Model not trained")
        
        probas = np.zeros((len(X), self.n_states))
        for t, x in enumerate(X):
            probas[t] = self._emission_prob(x)
        
        # Normalize
        probas /= np.sum(probas, axis=1, keepdims=True) + 1e-10
        return probas


class LiquiditySweepHMM:
    """
    Detects liquidity sweeps and stop-hunts using Hidden Markov Models.
    Classifies market regimes into: Normal, Sweep Accumulation, Stop Hunt.
    """
    
    # State definitions
    STATE_NORMAL = 0
    STATE_SWEEP_ACCUM = 1
    STATE_STOP_HUNT = 2
    
    def __init__(self, n_states: int = 3, model_path: str = 'models/liquidity_hmm.pkl',
                 sweep_threshold: float = 0.7, stop_hunt_threshold: float = 0.75):
        self.n_states = n_states
        self.model_path = Path(model_path)
        self.sweep_threshold = sweep_threshold
        self.stop_hunt_threshold = stop_hunt_threshold
        
        self.hmm = None
        self._feature_buffer = None
        self._state_history = None
        
        # Initialize feature buffer (window of observations)
        self._window_size = 100
        self._feature_dim = 6  # [returns, volume_ratio, wick_ratio, spread, momentum, volatility]
        self._feature_buffer = np.zeros((self._window_size, self._feature_dim), dtype=np.float32)
        self._state_history = np.zeros(self._window_size, dtype=np.int8)
        
        # Initialize or load HMM
        self._load_model()
        
        logger.info(f"LiquiditySweepHMM initialized with {n_states} states")
    
    def _load_model(self) -> None:
        """Load HMM from disk or initialize new one."""
        if self.model_path.exists():
            try:
                import pickle
                with open(self.model_path, 'rb') as f:
                    state = pickle.load(f)
                
                if HMMLEARN_AVAILABLE and state.get('use_hmmlearn', False):
                    self.hmm = hmm.GaussianHMM(n_components=self.n_states)
                    self.hmm.transmat_ = state['transmat']
                    self.hmm.means_ = state['means']
                    self.hmm.covars_ = state['covars']
                    self.hmm.startprob_ = state['startprob']
                else:
                    self.hmm = CustomHMM(n_states=self.n_states)
                    self.hmm.transmat_ = state['transmat']
                    self.hmm.means_ = state['means']
                    self.hmm.covars_ = state['covars']
                    self.hmm.startprob_ = state['startprob']
                    self.hmm._trained = True
                
                logger.info(f"Loaded HMM from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load HMM: {e}, initializing new model")
                self._init_model()
        else:
            self._init_model()
    
    def _init_model(self) -> None:
        """Initialize new HMM with default parameters."""
        if HMMLEARN_AVAILABLE:
            self.hmm = hmm.GaussianHMM(n_components=self.n_states, n_iter=100, random_state=42)
        else:
            self.hmm = CustomHMM(n_states=self.n_states, n_iter=100)
        logger.info("Initialized new HMM")
    
    def _extract_features(self, price_data: np.ndarray) -> np.ndarray:
        """
        Extract features for HMM from price/volume data.
        
        Features:
        0: Returns (price change %)
        1: Volume ratio (current / avg)
        2: Wick ratio (upper/lower wick)
        3: Spread (high-low range)
        4: Momentum (5-period returns)
        5: Volatility (rolling std)
        """
        if price_data.ndim == 1 or price_data.shape[1] < 4:
            return np.zeros(self._feature_dim, dtype=np.float32)
        
        # Extract OHLCV
        opens = price_data[:, 0]
        highs = price_data[:, 1]
        lows = price_data[:, 2]
        closes = price_data[:, 3]
        volumes = price_data[:, 4] if price_data.shape[1] > 4 else np.ones(len(closes))
        
        # Calculate features
        returns = np.diff(closes) / (closes[:-1] + 1e-10)
        returns = np.append(0, returns)
        
        avg_volume = np.mean(volumes[-20:]) + 1e-10
        volume_ratio = volumes / avg_volume
        
        # Wick calculations
        upper_wick = highs - np.maximum(opens, closes)
        lower_wick = np.minimum(opens, closes) - lows
        wick_ratio = upper_wick / (lower_wick + 1e-10)
        
        # Spread
        spread = (highs - lows) / (closes + 1e-10)
        
        # Momentum
        momentum = np.zeros_like(closes)
        momentum[5:] = (closes[5:] - closes[:-5]) / (closes[:-5] + 1e-10)
        
        # Volatility
        volatility = np.zeros_like(closes)
        for i in range(10, len(closes)):
            volatility[i] = np.std(returns[max(0, i-10):i+1])
        
        # Return latest feature vector
        features = np.array([
            returns[-1],
            volume_ratio[-1],
            wick_ratio[-1],
            spread[-1],
            momentum[-1],
            volatility[-1]
        ], dtype=np.float32)
        
        return features
    
    def classify(self, price_data: np.ndarray) -> Dict[str, float]:
        """
        Classify current market regime.
        
        Args:
            price_data: OHLCV data array
            
        Returns:
            Dictionary with classification probabilities and signals
        """
        # Extract features
        features = self._extract_features(price_data)
        
        # Update feature buffer
        self._feature_buffer[:-1] = self._feature_buffer[1:]
        self._feature_buffer[-1] = features
        
        sweep_prob = 0.0
        stop_hunt_prob = 0.0
        normal_prob = 0.0
        
        if hasattr(self.hmm, '_trained') and self.hmm._trained:
            try:
                # Get state probabilities
                probas = self.hmm.predict_proba(self._feature_buffer[-20:])
                
                # Aggregate probabilities over recent window
                recent_probas = probas[-5:]
                avg_probas = np.mean(recent_probas, axis=0)
                
                # Map states to probabilities
                if self.n_states >= 3:
                    normal_prob = avg_probas[self.STATE_NORMAL]
                    sweep_prob = avg_probas[self.STATE_SWEEP_ACCUM]
                    stop_hunt_prob = avg_probas[self.STATE_STOP_HUNT]
                elif self.n_states == 2:
                    normal_prob = avg_probas[0]
                    sweep_prob = avg_probas[1]
                    stop_hunt_prob = sweep_prob * 0.5
                else:
                    normal_prob = np.max(avg_probas)
                
                # Update state history
                current_state = np.argmax(avg_probas)
                self._state_history[:-1] = self._state_history[1:]
                self._state_history[-1] = current_state
                
            except Exception as e:
                logger.error(f"HMM classification failed: {e}")
                sweep_prob, stop_hunt_prob = self._heuristic_classification(features)
        else:
            # Use heuristic fallback
            sweep_prob, stop_hunt_prob = self._heuristic_classification(features)
            normal_prob = 1.0 - sweep_prob - stop_hunt_prob
        
        # Detect specific patterns
        is_sweep = sweep_prob > self.sweep_threshold
        is_stop_hunt = stop_hunt_prob > self.stop_hunt_threshold
        is_fake_breakout = self._detect_fake_breakout(price_data, sweep_prob)
        
        return {
            'sweep_prob': float(sweep_prob),
            'stop_hunt_prob': float(stop_hunt_prob),
            'normal_prob': float(normal_prob),
            'is_sweep': is_sweep,
            'is_stop_hunt': is_stop_hunt,
            'is_fake_breakout': is_fake_breakout,
            'current_state': int(self._state_history[-1]),
            'confidence': float(max(sweep_prob, stop_hunt_prob, normal_prob))
        }
    
    def _heuristic_classification(self, features: np.ndarray) -> Tuple[float, float]:
        """
        Heuristic classification when HMM not available.
        Uses rule-based detection of sweep and stop-hunt patterns.
        """
        returns, vol_ratio, wick_ratio, spread, momentum, volatility = features
        
        # Sweep detection: High volume + long wick + reversal
        sweep_score = 0.0
        if vol_ratio > 2.0:  # Volume spike
            sweep_score += 0.3
        if wick_ratio > 3.0 or wick_ratio < 0.33:  # Long wick
            sweep_score += 0.3
        if abs(returns) > volatility * 2:  # Large move
            sweep_score += 0.2
        if abs(momentum) < abs(returns) * 0.5:  # Reversal
            sweep_score += 0.2
        
        # Stop hunt detection: Extreme move + immediate reversal pattern
        stop_hunt_score = 0.0
        if spread > np.mean([spread, volatility]) * 3:  # Wide range
            stop_hunt_score += 0.3
        if abs(returns) > np.percentile([returns, volatility], 90):  # Extreme move
            stop_hunt_score += 0.3
        if vol_ratio > 3.0:  # Climactic volume
            stop_hunt_score += 0.2
        if wick_ratio > 5.0 or wick_ratio < 0.2:  # Very long single wick
            stop_hunt_score += 0.2
        
        return min(1.0, sweep_score), min(1.0, stop_hunt_score)
    
    def _detect_fake_breakout(self, price_data: np.ndarray, sweep_prob: float) -> bool:
        """
        Detect fake retail breakouts that lack institutional support.
        """
        if len(price_data) < 20 or price_data.ndim < 2 or price_data.shape[1] < 4:
            return False
        
        closes = price_data[:, 3]
        highs = price_data[:, 1]
        lows = price_data[:, 2]
        
        # Check for breakout above recent high
        recent_high = np.max(highs[-20:-1])
        current_high = highs[-1]
        
        if current_high > recent_high * 1.001:  # Breakout
            # Check if close is back inside range (fake breakout)
            if closes[-1] < recent_high:
                return True
        
        # Check for breakdown below recent low
        recent_low = np.min(lows[-20:-1])
        current_low = lows[-1]
        
        if current_low < recent_low * 0.999:  # Breakdown
            # Check if close is back inside range (fake breakout)
            if closes[-1] > recent_low:
                return True
        
        return False
    
    def warmup(self, historical_data: np.ndarray) -> None:
        """Warm up detector with historical data."""
        if len(historical_data) < self._window_size:
            return
        
        # Extract features for entire history
        features_list = []
        for i in range(len(historical_data)):
            feat = self._extract_features(historical_data[:i+1])
            features_list.append(feat)
        
        features_array = np.array(features_list, dtype=np.float32)
        
        # Train HMM if not already trained
        if not hasattr(self.hmm, '_trained') or not self.hmm._trained:
            try:
                self.hmm.fit(features_array)
                logger.info("HMM trained on historical data")
            except Exception as e:
                logger.error(f"HMM training failed: {e}")
        
        # Update buffer with recent data
        self._feature_buffer = features_array[-self._window_size:]
        
        # Classify historical data to populate state history
        if hasattr(self.hmm, 'predict'):
            try:
                states = self.hmm.predict(self._feature_buffer)
                self._state_history = states
            except Exception:
                pass
    
    def save(self) -> Dict[str, Any]:
        """Save HMM state to dictionary."""
        if self.hmm is None:
            return {}
        
        return {
            'transmat': getattr(self.hmm, 'transmat_', None),
            'means': getattr(self.hmm, 'means_', None),
            'covars': getattr(self.hmm, 'covars_', None),
            'startprob': getattr(self.hmm, 'startprob_', None),
            'use_hmmlearn': HMMLEARN_AVAILABLE,
            'n_states': self.n_states,
            'sweep_threshold': self.sweep_threshold,
            'stop_hunt_threshold': self.stop_hunt_threshold
        }
    
    def load(self, state: Dict[str, Any]) -> None:
        """Load HMM state from dictionary."""
        self.n_states = state.get('n_states', self.n_states)
        self.sweep_threshold = state.get('sweep_threshold', self.sweep_threshold)
        self.stop_hunt_threshold = state.get('stop_hunt_threshold', self.stop_hunt_threshold)
        
        if state.get('transmat') is not None:
            self._init_model()
            self.hmm.transmat_ = state['transmat']
            self.hmm.means_ = state['means']
            self.hmm.covars_ = state['covars']
            self.hmm.startprob_ = state['startprob']
            if hasattr(self.hmm, '_trained'):
                self.hmm._trained = True


__all__ = ['LiquiditySweepHMM', 'CustomHMM']
