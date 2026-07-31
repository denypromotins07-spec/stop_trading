"""
Greeks and Skew Signals from Volatility Surface.
Extracts Gamma and Delta skew anomalies to generate directional alpha signals.
Detects market maker positioning for volatility expansion prediction.
Strictly NumPy/SciPy based - no Pandas in hot path.
"""

import numpy as np
from scipy import stats
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    """Signal direction types."""
    BULLISH = 1
    BEARISH = -1
    NEUTRAL = 0


@dataclass
class GreeksResult:
    """Calculated Greeks from options chain."""
    delta: np.ndarray
    gamma: np.ndarray
    vega: np.ndarray
    theta: np.ndarray
    strike: np.ndarray
    expiry: float


@dataclass
class SkewSignal:
    """Volatility skew trading signal."""
    timestamp_ns: int
    asset: str
    signal_type: SignalType
    skew_value: float
    z_score: float
    confidence: float
    expected_move_pct: float
    gamma_exposure: float


class OptionsGreeksCalculator:
    """
    Calculates Black-Scholes Greeks for options chains.
    Optimized for vectorized computation.
    """
    
    def __init__(self, risk_free_rate: float = 0.05):
        """
        Args:
            risk_free_rate: Annual risk-free rate
        """
        self.r = risk_free_rate
    
    def _norm_cdf(self, x: np.ndarray) -> np.ndarray:
        """Standard normal CDF."""
        return 0.5 * (1 + np.erf(x / np.sqrt(2)))
    
    def _norm_pdf(self, x: np.ndarray) -> np.ndarray:
        """Standard normal PDF."""
        return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)
    
    def calculate_d1_d2(self, 
                        S: float, K: np.ndarray, T: float,
                        sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate d1 and d2 for Black-Scholes.
        
        Args:
            S: Spot price
            K: Strike array
            T: Time to expiry
            sigma: Implied volatility array
            
        Returns:
            Tuple of (d1, d2) arrays
        """
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (self.r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        
        return d1, d2
    
    def calculate_greeks(self,
                         S: float,
                         strikes: np.ndarray,
                         implied_vols: np.ndarray,
                         tenor: float,
                         option_type: str = 'call') -> GreeksResult:
        """
        Calculate full set of Greeks for options chain.
        
        Args:
            S: Spot price
            strikes: Strike array
            implied_vols: Implied volatilities
            tenor: Time to expiry
            option_type: 'call' or 'put'
            
        Returns:
            GreeksResult with all calculated values
        """
        sigma = implied_vols
        K = strikes
        T = tenor
        
        d1, d2 = self.calculate_d1_d2(S, K, T, sigma)
        
        # Common terms
        sqrt_T = np.sqrt(T)
        pdf_d1 = self._norm_pdf(d1)
        cdf_d1 = self._norm_cdf(d1)
        cdf_d2 = self._norm_cdf(d2)
        
        if option_type == 'call':
            delta = np.exp(-0.0 * T) * cdf_d1  # Assume no dividend
            gamma = np.exp(-0.0 * T) * pdf_d1 / (S * sigma * sqrt_T)
            vega = S * np.exp(-0.0 * T) * pdf_d1 * sqrt_T / 100  # Per 1% vol change
            theta = (-S * np.exp(-0.0 * T) * pdf_d1 * sigma / (2 * sqrt_T)
                    - self.r * K * np.exp(-self.r * T) * cdf_d2) / 365  # Daily
        else:  # put
            delta = np.exp(-0.0 * T) * (cdf_d1 - 1)
            gamma = np.exp(-0.0 * T) * pdf_d1 / (S * sigma * sqrt_T)
            vega = S * np.exp(-0.0 * T) * pdf_d1 * sqrt_T / 100
            theta = (-S * np.exp(-0.0 * T) * pdf_d1 * sigma / (2 * sqrt_T)
                    + self.r * K * np.exp(-self.r * T) * (1 - cdf_d2)) / 365
        
        return GreeksResult(
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            strike=strikes,
            expiry=tenor
        )
    
    def calculate_total_gamma(self,
                              S: float,
                              strikes: np.ndarray,
                              implied_vols: np.ndarray,
                              tenor: float,
                              open_interest: np.ndarray) -> float:
        """
        Calculate total gamma exposure weighted by open interest.
        
        Args:
            S: Spot price
            strikes: Strike array
            implied_vols: Implied volatilities
            tenor: Time to expiry
            open_interest: Open interest at each strike
            
        Returns:
            Total gamma exposure
        """
        greeks = self.calculate_greeks(S, strikes, implied_vols, tenor)
        
        # Gamma * Open Interest gives gamma exposure
        total_gamma = np.sum(greeks.gamma * open_interest)
        
        return total_gamma


class VolatilitySkewAnalyzer:
    """
    Analyzes volatility skew for directional signals.
    Detects when skew indicates potential breakouts.
    """
    
    def __init__(self,
                 lookback_periods: int = 252,
                 zscore_threshold: float = 2.0):
        """
        Args:
            lookback_periods: Historical periods for normalization
            zscore_threshold: Z-score threshold for signals
        """
        self.lookback_periods = lookback_periods
        self.zscore_threshold = zscore_threshold
        
        # Historical skew storage
        self.skew_history = np.zeros(lookback_periods)
        self.hist_idx = 0
        
        # Current skew metrics
        self.current_skew = 0.0
        self.skew_mean = 0.0
        self.skew_std = 1.0
    
    def calculate_skew(self, 
                       strikes: np.ndarray,
                       implied_vols: np.ndarray,
                       spot: float) -> Tuple[float, float, float]:
        """
        Calculate volatility skew metrics.
        
        Args:
            strikes: Strike array
            implied_vols: Implied volatilities
            spot: Current spot price
            
        Returns:
            Tuple of (skew_25d, skew_10d, put_call_skew)
        """
        # Find OTM puts and calls
        moneyness = strikes / spot
        
        # 25-delta approximation (typically ~10% OTM)
        put_25d_mask = (moneyness < 0.90) & (moneyness > 0.80)
        call_25d_mask = (moneyness > 1.10) & (moneyness < 1.20)
        
        # 10-delta approximation (typically ~20% OTM)
        put_10d_mask = (moneyness < 0.80) & (moneyness > 0.70)
        call_10d_mask = (moneyness > 1.20) & (moneyness < 1.30)
        
        # Calculate average IV for each bucket
        iv_put_25d = np.mean(implied_vols[put_25d_mask]) if np.any(put_25d_mask) else np.nan
        iv_call_25d = np.mean(implied_vols[call_25d_mask]) if np.any(call_25d_mask) else np.nan
        iv_put_10d = np.mean(implied_vols[put_10d_mask]) if np.any(put_10d_mask) else np.nan
        iv_call_10d = np.mean(implied_vols[call_10d_mask]) if np.any(call_10d_mask) else np.nan
        
        # Skew calculations
        skew_25d = iv_put_25d - iv_call_25d if not (np.isnan(iv_put_25d) or np.isnan(iv_call_25d)) else 0.0
        skew_10d = iv_put_10d - iv_call_10d if not (np.isnan(iv_put_10d) or np.isnan(iv_call_10d)) else 0.0
        put_call_skew = np.mean(implied_vols[moneyness < 1]) - np.mean(implied_vols[moneyness > 1])
        
        return skew_25d, skew_10d, put_call_skew
    
    def update_skew_history(self, skew: float):
        """Update historical skew record."""
        idx = self.hist_idx % self.lookback_periods
        self.skew_history[idx] = skew
        self.hist_idx += 1
        
        # Update running statistics
        valid_count = min(self.hist_idx, self.lookback_periods)
        start_idx = max(0, self.hist_idx - self.lookback_periods)
        recent_skews = self.skew_history[start_idx:self.hist_idx]
        
        self.skew_mean = np.mean(recent_skews)
        self.skew_std = np.std(recent_skews) + 1e-10
        self.current_skew = skew
    
    def get_skew_zscore(self, current_skew: float) -> float:
        """Calculate Z-score of current skew."""
        return (current_skew - self.skew_mean) / self.skew_std
    
    def generate_signal(self, 
                        asset: str,
                        skew_25d: float,
                        gamma_exposure: float,
                        spot: float) -> Optional[SkewSignal]:
        """
        Generate trading signal from skew analysis.
        
        Args:
            asset: Asset identifier
            skew_25d: 25-delta skew
            gamma_exposure: Total gamma exposure
            spot: Current spot price
            
        Returns:
            SkewSignal or None
        """
        import time
        timestamp_ns = time.time_ns()
        
        # Update history
        self.update_skew_history(skew_25d)
        
        # Calculate Z-score
        z_score = self.get_skew_zscore(skew_25d)
        
        # Determine signal type based on skew and gamma
        if abs(z_score) < self.zscore_threshold:
            signal_type = SignalType.NEUTRAL
            confidence = abs(z_score) / self.zscore_threshold
        elif z_score > self.zscore_threshold:
            # High put skew = bearish sentiment, potential reversal
            if gamma_exposure < 0:  # Negative gamma = unstable
                signal_type = SignalType.BEARISH
                confidence = min(abs(z_score) / 3.0, 1.0)
            else:
                signal_type = SignalType.BULLISH  # Contrarian
                confidence = min(abs(z_score) / 3.0, 0.8)
        else:  # z_score < -threshold
            # High call skew = bullish sentiment
            if gamma_exposure < 0:
                signal_type = SignalType.BULLISH
                confidence = min(abs(z_score) / 3.0, 1.0)
            else:
                signal_type = SignalType.BEARISH  # Contrarian
                confidence = min(abs(z_score) / 3.0, 0.8)
        
        # Estimate expected move from skew
        expected_move_pct = abs(skew_25d) * 100  # Rough approximation
        
        return SkewSignal(
            timestamp_ns=timestamp_ns,
            asset=asset,
            signal_type=signal_type,
            skew_value=skew_25d,
            z_score=z_score,
            confidence=confidence,
            expected_move_pct=expected_move_pct,
            gamma_exposure=gamma_exposure
        )


class GammaExposureMonitor:
    """
    Monitors dealer gamma exposure for volatility regime detection.
    Negative gamma regimes indicate potential volatility expansion.
    """
    
    def __init__(self, assets: List[str]):
        """
        Args:
            assets: List of assets to monitor
        """
        self.assets = assets
        self.greeks_calc = OptionsGreeksCalculator()
        
        # Gamma exposure tracking
        self.gamma_levels = {asset: [] for asset in assets}
        self.max_history = 100
        
        # Regime classification thresholds
        self.gamma_threshold_low = -1e6   # Very negative gamma
        self.gamma_threshold_high = 1e6   # Very positive gamma
    
    def update_gamma_exposure(self,
                              asset: str,
                              spot: float,
                              strikes: np.ndarray,
                              implied_vols: np.ndarray,
                              tenor: float,
                              open_interest: np.ndarray) -> float:
        """
        Update gamma exposure for an asset.
        
        Args:
            asset: Asset identifier
            spot: Current spot price
            strikes: Strike array
            implied_vols: Implied volatilities
            tenor: Time to expiry
            open_interest: Open interest at each strike
            
        Returns:
            Total gamma exposure
        """
        total_gamma = self.greeks_calc.calculate_total_gamma(
            spot, strikes, implied_vols, tenor, open_interest
        )
        
        # Store in history
        gamma_list = self.gamma_levels[asset]
        gamma_list.append(total_gamma)
        if len(gamma_list) > self.max_history:
            gamma_list.pop(0)
        
        return total_gamma
    
    def get_regime(self, asset: str) -> Tuple[str, float]:
        """
        Get current gamma regime for an asset.
        
        Returns:
            Tuple of (regime_name, gamma_level)
        """
        if not self.gamma_levels[asset]:
            return "UNKNOWN", 0.0
        
        current_gamma = self.gamma_levels[asset][-1]
        
        if current_gamma < self.gamma_threshold_low:
            regime = "NEGATIVE_GAMMA"  # Volatility expansion likely
        elif current_gamma > self.gamma_threshold_high:
            regime = "POSITIVE_GAMMA"  # Mean-reverting, low vol
        else:
            regime = "NEUTRAL"
        
        return regime, current_gamma
    
    def detect_gamma_flip(self, asset: str) -> Optional[Dict]:
        """
        Detect if gamma has recently flipped sign.
        
        Args:
            asset: Asset identifier
            
        Returns:
            Signal dictionary if flip detected, None otherwise
        """
        gamma_history = self.gamma_levels[asset]
        
        if len(gamma_history) < 3:
            return None
        
        current = gamma_history[-1]
        previous = gamma_history[-2]
        
        # Check for sign flip
        if current * previous < 0:  # Signs are different
            import time
            return {
                'asset': asset,
                'flip_direction': 'positive' if current > 0 else 'negative',
                'previous_gamma': previous,
                'current_gamma': current,
                'timestamp_ns': time.time_ns(),
                'action': 'Adjust execution aggression based on new regime'
            }
        
        return None


def generate_volatility_signals(asset: str,
                                spot: float,
                                strikes: np.ndarray,
                                implied_vols: np.ndarray,
                                tenor: float,
                                open_interest: np.ndarray,
                                skew_analyzer: VolatilitySkewAnalyzer,
                                gamma_monitor: GammaExposureMonitor) -> List[Dict]:
    """
    Generate comprehensive volatility-based signals.
    
    Args:
        asset: Asset identifier
        spot: Current spot price
        strikes: Strike array
        implied_vols: Implied volatilities
        tenor: Time to expiry
        open_interest: Open interest
        skew_analyzer: VolatilitySkewAnalyzer instance
        gamma_monitor: GammaExposureMonitor instance
        
    Returns:
        List of signal dictionaries
    """
    signals = []
    
    # Calculate skew
    skew_25d, skew_10d, pc_skew = skew_analyzer.calculate_skew(strikes, implied_vols, spot)
    
    # Update gamma exposure
    gamma_exp = gamma_monitor.update_gamma_exposure(
        asset, spot, strikes, implied_vols, tenor, open_interest
    )
    
    # Get gamma regime
    regime, _ = gamma_monitor.get_regime(asset)
    
    # Generate skew signal
    skew_signal = skew_analyzer.generate_signal(asset, skew_25d, gamma_exp, spot)
    if skew_signal is not None:
        signals.append({
            'type': 'volatility_skew',
            'asset': asset,
            'direction': skew_signal.signal_type.name,
            'skew_25d': skew_25d,
            'z_score': skew_signal.z_score,
            'confidence': skew_signal.confidence,
            'gamma_regime': regime,
            'gamma_exposure': gamma_exp,
            'expected_move_pct': skew_signal.expected_move_pct
        })
    
    # Check for gamma flip
    flip_signal = gamma_monitor.detect_gamma_flip(asset)
    if flip_signal is not None:
        signals.append({
            'type': 'gamma_flip',
            **flip_signal
        })
    
    return signals


__all__ = [
    'OptionsGreeksCalculator',
    'VolatilitySkewAnalyzer',
    'GammaExposureMonitor',
    'GreeksResult',
    'SkewSignal',
    'SignalType',
    'generate_volatility_signals'
]
