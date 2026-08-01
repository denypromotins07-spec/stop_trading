"""
Gaussian Mixture Model (GMM) and lightweight HMM for macro regime classification.
Classifies regimes: Risk-On, Risk-Off, Stagflation.
Dynamically shifts bot's global beta exposure based on detected state.
"""

import numpy as np
from numba import njit
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass
import threading
from enum import IntEnum


class RegimeState(IntEnum):
    """Macro-economic regime states."""
    RISK_ON = 0      # Growth up, volatility down, credit spreads tight
    RISK_OFF = 1     # Growth down, volatility up, flight to safety
    STAGFLATION = 2  # High inflation, low growth, high rates
    TRANSITION = 3   # Uncertain/transitioning state


@njit(cache=True)
def gaussian_pdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """Compute multivariate Gaussian PDF."""
    n = len(x)
    diff = x - mean
    
    # Compute determinant and inverse
    try:
        det = np.linalg.det(cov)
        if det < 1e-10:
            return 1e-10
        
        cov_inv = np.linalg.inv(cov)
        
        # Mahalanobis distance
        mahalanobis = diff @ cov_inv @ diff
        
        # PDF
        norm_const = 1.0 / (np.power(2 * np.pi, n / 2) * np.sqrt(det))
        pdf = norm_const * np.exp(-0.5 * mahalanobis)
        
        return max(pdf, 1e-10)
    except:
        return 1e-10


@njit(cache=True)
def gmm_e_step(
    data: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
    weights: np.ndarray,
    n_components: int
) -> np.ndarray:
    """E-step: compute responsibilities."""
    n_samples = data.shape[0]
    responsibilities = np.zeros((n_samples, n_components))
    
    for k in range(n_components):
        for i in range(n_samples):
            responsibilities[i, k] = weights[k] * gaussian_pdf(
                data[i], means[k], covs[k]
            )
    
    # Normalize
    for i in range(n_samples):
        total = np.sum(responsibilities[i])
        if total > 1e-10:
            responsibilities[i] /= total
        else:
            responsibilities[i] = 1.0 / n_components
    
    return responsibilities


@njit(cache=True)
def gmm_m_step(
    data: np.ndarray,
    responsibilities: np.ndarray,
    n_components: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """M-step: update parameters."""
    n_samples, n_features = data.shape
    
    # Update weights
    Nk = np.sum(responsibilities, axis=0)
    weights = Nk / n_samples
    
    # Update means
    means = np.zeros((n_components, n_features))
    for k in range(n_components):
        for j in range(n_features):
            weighted_sum = 0.0
            for i in range(n_samples):
                weighted_sum += responsibilities[i, k] * data[i, j]
            
            if Nk[k] > 1e-10:
                means[k, j] = weighted_sum / Nk[k]
    
    # Update covariances with regularization
    covs = np.zeros((n_components, n_features, n_features))
    for k in range(n_components):
        for i in range(n_features):
            for j in range(n_features):
                weighted_sum = 0.0
                for s in range(n_samples):
                    diff_i = data[s, i] - means[k, i]
                    diff_j = data[s, j] - means[k, j]
                    weighted_sum += responsibilities[s, k] * diff_i * diff_j
                
                if Nk[k] > 1e-10:
                    covs[k, i, j] = weighted_sum / Nk[k]
                
                # Add regularization to diagonal
                if i == j:
                    covs[k, i, j] += 1e-6
    
    return means, covs, weights


@njit(cache=True)
def viterbi_decode(
    log_probs: np.ndarray,
    trans_matrix: np.ndarray,
    pi: np.ndarray,
    n_states: int
) -> np.ndarray:
    """Viterbi algorithm for HMM decoding."""
    n_obs = log_probs.shape[0]
    
    # Initialize
    delta = np.zeros((n_obs, n_states))
    psi = np.zeros((n_obs, n_states), dtype=np.int32)
    
    delta[0] = np.log(pi + 1e-10) + log_probs[0]
    
    # Forward pass
    for t in range(1, n_obs):
        for j in range(n_states):
            max_val = -np.inf
            max_idx = 0
            for i in range(n_states):
                val = delta[t-1, i] + np.log(trans_matrix[i, j] + 1e-10)
                if val > max_val:
                    max_val = val
                    max_idx = i
            
            delta[t, j] = max_val + log_probs[t, j]
            psi[t, j] = max_idx
    
    # Backtrack
    path = np.zeros(n_obs, dtype=np.int32)
    path[n_obs - 1] = np.argmax(delta[n_obs - 1])
    
    for t in range(n_obs - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    
    return path


@dataclass
class RegimeResult:
    """Result of regime classification."""
    current_regime: RegimeState
    regime_probabilities: np.ndarray
    confidence: float
    suggested_beta: float  # Recommended portfolio beta
    regime_description: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_regime": self.current_regime.name,
            "regime_probabilities": self.regime_probabilities.tolist(),
            "confidence": self.confidence,
            "suggested_beta": self.suggested_beta,
            "regime_description": self.regime_description
        }


class MacroRegimeClassifier:
    """
    GMM-HMM hybrid for macro regime classification.
    Uses GMM for emission probabilities and HMM for temporal smoothing.
    """
    
    # Feature indices
    VOLATILITY = 0
    GROWTH = 1
    INFLATION = 2
    CREDIT_SPREAD = 3
    
    def __init__(
        self,
        n_components: int = 3,
        n_features: int = 4,
        history_window: int = 50
    ):
        self.n_components = n_components
        self.n_features = n_features
        self.history_window = history_window
        
        # GMM parameters (initialized heuristically)
        self._means = self._initialize_means()
        self._covs = self._initialize_covariances()
        self._weights = np.ones(n_components) / n_components
        
        # HMM transition matrix (persistent regimes more likely)
        self._trans_matrix = np.array([
            [0.85, 0.10, 0.05],  # Risk-On tends to persist
            [0.08, 0.85, 0.07],  # Risk-Off tends to persist
            [0.10, 0.10, 0.80]   # Stagflation moderately persistent
        ])
        
        # Initial state distribution
        self._pi = np.array([0.5, 0.35, 0.15])
        
        # Observation history for HMM
        self._observation_history: List[np.ndarray] = []
        
        # State tracking
        self._current_regime = RegimeState.TRANSITION
        self._regime_history: List[RegimeState] = []
        
        # Beta mapping (recommended portfolio exposure)
        self._beta_map = {
            RegimeState.RISK_ON: 1.2,       # Overweight risk
            RegimeState.RISK_OFF: 0.3,      # Underweight risk
            RegimeState.STAGFLATION: 0.5,   # Defensive positioning
            RegimeState.TRANSITION: 0.7     # Neutral-cautious
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Training state
        self._is_trained = False
        self._training_samples = 0
    
    def _initialize_means(self) -> np.ndarray:
        """Initialize GMM means for each regime."""
        # [volatility, growth, inflation, credit_spread]
        return np.array([
            [0.1, 0.3, 0.02, 0.01],   # Risk-On: low vol, high growth, low inflation
            [0.4, -0.2, 0.03, 0.04],  # Risk-Off: high vol, low growth, moderate inflation
            [0.3, -0.1, 0.08, 0.03]   # Stagflation: high vol, low growth, high inflation
        ])
    
    def _initialize_covariances(self) -> np.ndarray:
        """Initialize GMM covariances."""
        covs = np.zeros((self.n_components, self.n_features, self.n_features))
        
        for k in range(self.n_components):
            # Diagonal covariance with regime-specific variance
            variances = np.array([0.05, 0.05, 0.01, 0.01])
            covs[k] = np.diag(variances)
        
        return covs
    
    def partial_fit(self, observation: np.ndarray) -> None:
        """
        Incrementally update model with new observation.
        Online learning for adaptive regime detection.
        """
        if len(observation) != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {len(observation)}")
        
        with self._lock:
            # Add to history
            self._observation_history.append(observation.copy())
            
            # Trim history
            while len(self._observation_history) > self.history_window:
                self._observation_history.pop(0)
            
            self._training_samples += 1
            
            # Retrain when enough samples accumulated
            if self._training_samples >= 20 and len(self._observation_history) >= 20:
                self._retrain_gmm()
                self._is_trained = True
    
    def _retrain_gmm(self) -> None:
        """Retrain GMM on recent observations."""
        if len(self._observation_history) < 20:
            return
        
        data = np.array(self._observation_history)
        
        # Run a few EM iterations
        for _ in range(5):
            responsibilities = gmm_e_step(
                data, self._means, self._covs, self._weights, self.n_components
            )
            self._means, self._covs, self._weights = gmm_m_step(
                data, responsibilities, self.n_components
            )
    
    def classify(self, observation: np.ndarray) -> RegimeResult:
        """
        Classify the current macro regime.
        Applies GMM for emission probs and HMM for smoothing.
        """
        with self._lock:
            # Get GMM emission probabilities
            emissions = self._compute_emissions(observation)
            
            # Apply HMM smoothing if we have history
            if len(self._observation_history) > 1:
                log_probs = np.log(emissions + 1e-10)
                
                # Build recent observation log-probs for Viterbi
                recent_log_probs = []
                for obs in self._observation_history[-10:]:
                    em = self._compute_emissions(obs)
                    recent_log_probs.append(np.log(em + 1e-10))
                
                if len(recent_log_probs) > 1:
                    log_probs_array = np.array(recent_log_probs)
                    path = viterbi_decode(
                        log_probs_array, self._trans_matrix, self._pi, self.n_components
                    )
                    
                    # Use final state from Viterbi
                    final_state = path[-1]
                    
                    # Get smoothed probabilities from last step
                    smoothed_probs = self._forward_algorithm(log_probs_array)
                    probs = smoothed_probs[-1]
                else:
                    final_state = np.argmax(emissions)
                    probs = emissions
            else:
                final_state = np.argmax(emissions)
                probs = emissions
            
            # Map to regime
            regime = RegimeState(final_state)
            
            # Update state
            self._current_regime = regime
            self._regime_history.append(regime)
            if len(self._regime_history) > 100:
                self._regime_history.pop(0)
            
            # Calculate confidence
            confidence = float(np.max(probs))
            
            # Get suggested beta
            suggested_beta = self._beta_map[regime]
            
            # Generate description
            description = self._get_regime_description(regime)
            
            return RegimeResult(
                current_regime=regime,
                regime_probabilities=probs,
                confidence=confidence,
                suggested_beta=suggested_beta,
                regime_description=description
            )
    
    def _compute_emissions(self, observation: np.ndarray) -> np.ndarray:
        """Compute GMM emission probabilities for observation."""
        emissions = np.zeros(self.n_components)
        
        for k in range(self.n_components):
            emissions[k] = gaussian_pdf(
                observation, self._means[k], self._covs[k]
            )
        
        # Weight by mixture weights
        emissions *= self._weights
        
        # Normalize
        total = np.sum(emissions)
        if total > 1e-10:
            emissions /= total
        
        return emissions
    
    def _forward_algorithm(self, log_probs: np.ndarray) -> np.ndarray:
        """Forward algorithm for HMM filtering."""
        n_obs = log_probs.shape[0]
        alpha = np.zeros((n_obs, self.n_components))
        
        # Initialize
        alpha[0] = np.log(self._pi + 1e-10) + log_probs[0]
        
        # Forward pass
        for t in range(1, n_obs):
            for j in range(self.n_components):
                log_sum = -np.inf
                for i in range(self.n_components):
                    val = alpha[t-1, i] + np.log(self._trans_matrix[i, j] + 1e-10)
                    log_sum = np.logaddexp(log_sum, val)
                alpha[t, j] = log_sum + log_probs[t, j]
        
        # Convert back to probabilities
        probs = np.zeros((n_obs, self.n_components))
        for t in range(n_obs):
            log_sum = np.max(alpha[t])
            probs[t] = np.exp(alpha[t] - log_sum)
            probs[t] /= np.sum(probs[t]) + 1e-10
        
        return probs
    
    def _get_regime_description(self, regime: RegimeState) -> str:
        """Get human-readable regime description."""
        descriptions = {
            RegimeState.RISK_ON: (
                "Risk-On: Favorable conditions for risk assets. "
                "Consider increasing beta exposure to equities and crypto."
            ),
            RegimeState.RISK_OFF: (
                "Risk-Off: Defensive posture recommended. "
                "Reduce beta, increase cash and safe-haven allocations."
            ),
            RegimeState.STAGFLATION: (
                "Stagflation: Challenging environment with high inflation and low growth. "
                "Consider commodities, TIPS, and defensive equity positions."
            ),
            RegimeState.TRANSITION: (
                "Transition: Regime uncertainty elevated. "
                "Maintain moderate beta with flexibility to adjust."
            )
        }
        return descriptions.get(regime, "Unknown regime")
    
    def get_current_regime(self) -> RegimeState:
        """Get current regime state."""
        return self._current_regime
    
    def get_regime_history(self) -> List[RegimeState]:
        """Get recent regime history."""
        return self._regime_history.copy()
    
    def is_trained(self) -> bool:
        """Check if model has been trained."""
        return self._is_trained
    
    def reset(self) -> None:
        """Reset all state."""
        with self._lock:
            self._observation_history.clear()
            self._regime_history.clear()
            self._current_regime = RegimeState.TRANSITION
            self._is_trained = False
            self._training_samples = 0
            
            # Reinitialize parameters
            self._means = self._initialize_means()
            self._covs = self._initialize_covariances()
            self._weights = np.ones(self.n_components) / self.n_components
    
    def to_dict(self) -> Dict[str, Any]:
        """Export state for serialization."""
        with self._lock:
            return {
                "current_regime": self._current_regime.name,
                "is_trained": self._is_trained,
                "training_samples": self._training_samples,
                "history_size": len(self._observation_history),
                "regime_history_length": len(self._regime_history),
                "beta_map": {k.name: v for k, v in self._beta_map.items()}
            }


# Global singleton instance
_regime_instance: Optional[MacroRegimeClassifier] = None
_instance_lock = threading.Lock()


def get_regime_classifier() -> MacroRegimeClassifier:
    """Get or create the global regime classifier."""
    global _regime_instance
    if _regime_instance is None:
        with _instance_lock:
            if _regime_instance is None:
                _regime_instance = MacroRegimeClassifier()
    return _regime_instance


if __name__ == "__main__":
    # Test the regime classifier
    print("Testing MacroRegimeClassifier:")
    
    classifier = MacroRegimeClassifier()
    
    # Simulate observations for different regimes
    np.random.seed(42)
    
    # Risk-On period
    print("\n--- Risk-On Period ---")
    for i in range(30):
        obs = np.array([
            0.1 + np.random.randn() * 0.02,   # Low volatility
            0.3 + np.random.randn() * 0.05,   # High growth
            0.02 + np.random.randn() * 0.005, # Low inflation
            0.01 + np.random.randn() * 0.002  # Tight spreads
        ])
        classifier.partial_fit(obs)
    
    result = classifier.classify(obs)
    print(f"Regime: {result.current_regime.name}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Suggested Beta: {result.suggested_beta}")
    
    # Risk-Off period
    print("\n--- Risk-Off Period ---")
    for i in range(30):
        obs = np.array([
            0.4 + np.random.randn() * 0.05,   # High volatility
            -0.2 + np.random.randn() * 0.05,  # Low growth
            0.03 + np.random.randn() * 0.005, # Moderate inflation
            0.04 + np.random.randn() * 0.01   # Wide spreads
        ])
        classifier.partial_fit(obs)
    
    result = classifier.classify(obs)
    print(f"Regime: {result.current_regime.name}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Suggested Beta: {result.suggested_beta}")
    
    # Stagflation period
    print("\n--- Stagflation Period ---")
    for i in range(30):
        obs = np.array([
            0.3 + np.random.randn() * 0.05,   # High volatility
            -0.1 + np.random.randn() * 0.05,  # Low growth
            0.08 + np.random.randn() * 0.01,  # High inflation
            0.03 + np.random.randn() * 0.005  # Moderate spreads
        ])
        classifier.partial_fit(obs)
    
    result = classifier.classify(obs)
    print(f"Regime: {result.current_regime.name}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Suggested Beta: {result.suggested_beta}")
    
    print(f"\nModel State: {classifier.to_dict()}")
