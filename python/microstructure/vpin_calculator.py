# Volume-Synchronized Probability of Informed Trading (VPIN) Calculator
# Vectorized implementation for toxicity feature generation

from __future__ import annotations
import logging
import numpy as np
from typing import Optional, Tuple

log = logging.getLogger(__name__)


class VPINCalculator:
    """
    Vectorized VPIN calculator for order flow toxicity measurement.
    Predicts adverse selection during high-frequency bursts.
    
    Uses Easley, Kiefer, O'Hara, and Paperman (2012) methodology.
    Optimized for HFT with pre-allocated buffers and zero-copy operations.
    """

    def __init__(
        self,
        bucket_size: int = 1000,  # Volume per bucket (in contracts)
        num_buckets: int = 50,    # Number of buckets for VPIN calculation
        tick_rule_threshold: float = 0.0001,  # Price change threshold for tick rule
    ) -> None:
        self.bucket_size = bucket_size
        self.num_buckets = num_buckets
        self.tick_rule_threshold = tick_rule_threshold
        
        # Pre-allocated buffers for volume buckets
        self._buy_volume = np.zeros(num_buckets, dtype=np.float64)
        self._sell_volume = np.zeros(num_buckets, dtype=np.float64)
        self._bucket_counts = np.zeros(num_buckets, dtype=np.int32)
        
        # Current bucket state
        self._current_bucket = 0
        self._current_buy_vol = 0.0
        self._current_sell_vol = 0.0
        self._prev_price: Optional[float] = None
        self._total_volume_processed = 0.0
        
        # VPIN history (circular buffer)
        self._vpin_history = np.zeros(100, dtype=np.float64)
        self._vpin_head = 0

    def _classify_trade(self, price: float, volume: float) -> Tuple[float, float]:
        """
        Classify trade as buy or sell using tick rule.
        Returns (buy_volume, sell_volume) tuple.
        """
        if self._prev_price is None:
            # First trade, split evenly
            return volume / 2, volume / 2
        
        price_change = price - self._prev_price
        
        if price_change > self.tick_rule_threshold:
            # Buy-initiated
            return volume, 0.0
        elif price_change < -self.tick_rule_threshold:
            # Sell-initiated
            return 0.0, volume
        else:
            # No price change, use previous classification
            return volume / 2, volume / 2

    def _fill_bucket(self, buy_vol: float, sell_vol: float) -> Optional[int]:
        """
        Fill current volume bucket.
        Returns bucket index if bucket is complete, None otherwise.
        """
        total_vol = buy_vol + sell_vol
        remaining_bucket = self.bucket_size - (self._current_buy_vol + self._current_sell_vol)
        
        if total_vol >= remaining_bucket:
            # Bucket filled
            fraction = remaining_bucket / total_vol if total_vol > 0 else 0.5
            
            self._current_buy_vol += buy_vol * fraction
            self._current_sell_vol += sell_vol * fraction
            
            # Store completed bucket
            self._buy_volume[self._current_bucket] = self._current_buy_vol
            self._sell_volume[self._current_bucket] = self._current_sell_vol
            self._bucket_counts[self._current_bucket] = 1
            
            completed_bucket = self._current_bucket
            
            # Move to next bucket
            self._current_bucket = (self._current_bucket + 1) % self.num_buckets
            self._current_buy_vol = buy_vol * (1 - fraction)
            self._current_sell_vol = sell_vol * (1 - fraction)
            
            return completed_bucket
        else:
            # Bucket not yet filled
            self._current_buy_vol += buy_vol
            self._current_sell_vol += sell_vol
            return None

    def compute_vpin(self) -> float:
        """
        Compute VPIN from filled buckets.
        VPIN = sum(|buy_vol - sell_vol|) / sum(buy_vol + sell_vol)
        """
        filled_buckets = np.sum(self._bucket_counts > 0)
        
        if filled_buckets < self.num_buckets * 0.5:
            # Not enough data
            return 0.0
        
        buy_total = np.sum(self._buy_volume)
        sell_total = np.sum(self._sell_volume)
        
        if buy_total + sell_total == 0:
            return 0.0
        
        abs_imbalance = np.sum(np.abs(self._buy_volume - self._sell_volume))
        vpin = abs_imbalance / (buy_total + sell_total)
        
        return min(vpin, 1.0)  # Cap at 1.0

    def update(self, prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """
        Process batch of trades and return VPIN values.
        Uses vectorized operations for efficiency.
        
        Args:
            prices: Array of trade prices
            volumes: Array of trade volumes
            
        Returns:
            Array of VPIN values (one per input trade)
        """
        n_trades = len(prices)
        vpin_output = np.zeros(n_trades, dtype=np.float64)
        
        # Vectorized tick rule classification
        if self._prev_price is not None:
            price_changes = np.diff(np.concatenate([[self._prev_price], prices]))
        else:
            price_changes = np.diff(prices)
            if len(price_changes) > 0:
                self._prev_price = prices[0]
        
        # Classify trades
        buy_volumes = np.where(price_changes > self.tick_rule_threshold, volumes[1:], 
                               np.where(price_changes < -self.tick_rule_threshold, 0.0, volumes[1:] / 2))
        sell_volumes = np.where(price_changes < -self.tick_rule_threshold, volumes[1:],
                                np.where(price_changes > self.tick_rule_threshold, 0.0, volumes[1:] / 2))
        
        # Handle first trade
        if self._prev_price is None and len(prices) > 0:
            buy_volumes = np.concatenate([[volumes[0] / 2], buy_volumes])
            sell_volumes = np.concatenate([[volumes[0] / 2], sell_volumes])
            self._prev_price = prices[-1]
        
        # Process each trade
        for i in range(n_trades):
            bv = buy_volumes[i] if i < len(buy_volumes) else volumes[i] / 2
            sv = sell_volumes[i] if i < len(sell_volumes) else volumes[i] / 2
            
            self._fill_bucket(bv, sv)
            self._total_volume_processed += bv + sv
            
            # Compute and store VPIN
            vpin = self.compute_vpin()
            vpin_output[i] = vpin
            
            # Store in history buffer
            self._vpin_history[self._vpin_head] = vpin
            self._vpin_head = (self._vpin_head + 1) % 100
        
        return vpin_output

    def get_toxicity_features(self) -> dict[str, float]:
        """
        Generate toxicity features for ML models.
        """
        current_vpin = self.compute_vpin()
        
        # Get recent VPIN statistics
        valid_vpins = self._vpin_history[:max(1, self._vpin_head)]
        if len(valid_vpins) < 10:
            valid_vpins = self._vpin_history[self._vpin_history > 0]
        
        if len(valid_vpins) == 0:
            valid_vpins = np.array([0.0])
        
        return {
            "vpin_current": current_vpin,
            "vpin_mean": float(np.mean(valid_vpins)),
            "vpin_std": float(np.std(valid_vpins)),
            "vpin_max": float(np.max(valid_vpins)),
            "vpin_min": float(np.min(valid_vpins)),
            "vpin_percentile_90": float(np.percentile(valid_vpins, 90)),
            "toxicity_regime": self._classify_toxicity_regime(current_vpin),
            "volume_imbalance": float((self._current_buy_vol - self._current_sell_vol) / 
                                      (self._current_buy_vol + self._current_sell_vol + 1e-9)),
        }

    def _classify_toxicity_regime(self, vpin: float) -> int:
        """
        Classify toxicity regime.
        0: Low toxicity, 1: Medium, 2: High, 3: Extreme
        """
        if vpin < 0.2:
            return 0
        elif vpin < 0.4:
            return 1
        elif vpin < 0.6:
            return 2
        else:
            return 3

    def reset(self) -> None:
        """Reset all state."""
        self._buy_volume.fill(0.0)
        self._sell_volume.fill(0.0)
        self._bucket_counts.fill(0)
        self._current_bucket = 0
        self._current_buy_vol = 0.0
        self._current_sell_vol = 0.0
        self._prev_price = None
        self._vpin_history.fill(0.0)
        self._vpin_head = 0
        log.info("VPINCalculator reset")
