"""
Advanced Execution Sniper Algorithm - Iceberg Detection & Dark Pool Front-Running
Detects hidden liquidity via L3 trade tick anomalies using pure numpy math.
Front-runs replenishment of dark pools to capture institutional size before market reprices.
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class SniperSignal:
    signal_type: SignalType
    confidence: float
    estimated_iceberg_size: float
    price_level: float
    urgency_score: float


class IcebergSniper:
    """
    Detects iceberg orders by analyzing L3 trade tick anomalies.
    Uses statistical deviation in trade size, timing, and price impact.
    """

    def __init__(
        self,
        lookback_window: int = 50,
        size_threshold_std: float = 2.5,
        replenishment_rate_threshold: float = 0.7,
        min_iceberg_size: float = 100.0,
    ):
        self.lookback_window = lookback_window
        self.size_threshold_std = size_threshold_std
        self.replenishment_rate_threshold = replenishment_rate_threshold
        self.min_iceberg_size = min_iceberg_size

        # Rolling buffers for L3 data
        self.trade_sizes: np.ndarray = np.zeros(lookback_window, dtype=np.float64)
        self.trade_prices: np.ndarray = np.zeros(lookback_window, dtype=np.float64)
        self.trade_timestamps: np.ndarray = np.zeros(lookback_window, dtype=np.int64)
        self.aggressor_flags: np.ndarray = np.zeros(lookback_window, dtype=np.int8)
        self.buffer_idx: int = 0
        self.buffer_filled: bool = False

        # Welford online statistics for trade sizes
        self.mean_size: float = 0.0
        self.m2_size: float = 0.0
        self.count: int = 0

        # Replenishment tracking
        self.last_price_levels: dict = {}
        self.replenishment_counts: dict = {}

    def _update_welford(self, size: float) -> None:
        """Update Welford online statistics for trade size."""
        self.count += 1
        delta = size - self.mean_size
        self.mean_size += delta / self.count
        delta2 = size - self.mean_size
        self.m2_size += delta * delta2

    def _get_size_std(self) -> float:
        """Calculate standard deviation from Welford statistics."""
        if self.count < 2:
            return 0.0
        variance = self.m2_size / (self.count - 1)
        return np.sqrt(variance)

    def _push_tick(
        self,
        size: float,
        price: float,
        timestamp_ns: int,
        is_buyer_aggressor: bool,
    ) -> None:
        """Push L3 tick data into rolling buffer."""
        self.trade_sizes[self.buffer_idx] = size
        self.trade_prices[self.buffer_idx] = price
        self.trade_timestamps[self.buffer_idx] = timestamp_ns
        self.aggressor_flags[self.buffer_idx] = 1 if is_buyer_aggressor else 0

        self._update_welford(size)

        self.buffer_idx = (self.buffer_idx + 1) % self.lookback_window
        if self.buffer_idx == 0:
            self.buffer_filled = True

    def _detect_size_anomaly(self) -> Tuple[bool, float]:
        """
        Detect anomalous trade sizes indicating iceberg activity.
        Returns (is_anomaly, z_score).
        """
        if not self.buffer_filled:
            return False, 0.0

        current_size = self.trade_sizes[(self.buffer_idx - 1) % self.lookback_window]
        std_dev = self._get_size_std()

        if std_dev < 1e-9:
            return False, 0.0

        z_score = (current_size - self.mean_size) / std_dev
        is_anomaly = abs(z_score) > self.size_threshold_std

        return is_anomaly, z_score

    def _detect_replenishment(self, price: float, side: str) -> float:
        """
        Detect rapid replenishment at a price level indicating dark pool activity.
        Returns replenishment rate (0.0 to 1.0).
        """
        key = f"{price}_{side}"
        current_time = self.trade_timestamps[(self.buffer_idx - 1) % self.lookback_window]

        if key not in self.last_price_levels:
            self.last_price_levels[key] = current_time
            self.replenishment_counts[key] = 0
            return 0.0

        last_time = self.last_price_levels[key]
        time_delta_ns = current_time - last_time

        # Replenishment within 100ms indicates aggressive institutional activity
        if time_delta_ns < 100_000_000:  # 100ms in nanoseconds
            self.replenishment_counts[key] = min(
                self.replenishment_counts.get(key, 0) + 1, 10
            )
        else:
            self.replenishment_counts[key] = max(
                self.replenishment_counts.get(key, 0) - 1, 0
            )

        self.last_price_levels[key] = current_time
        replenishment_rate = self.replenishment_counts[key] / 10.0

        return replenishment_rate

    def _calculate_price_impact(self, is_buyer_aggressor: bool) -> float:
        """
        Calculate price impact anomaly - iceberg orders often show low price impact
        despite large size due to hidden liquidity absorption.
        """
        if self.buffer_idx < 5:
            return 0.0

        recent_sizes = self.trade_sizes[: self.buffer_idx]
        recent_prices = self.trade_prices[: self.buffer_idx]
        recent_aggressors = self.aggressor_flags[: self.buffer_idx]

        # Filter for same-side aggressor trades
        mask = recent_aggressors == (1 if is_buyer_aggressor else 0)
        if np.sum(mask) < 3:
            return 0.0

        same_side_prices = recent_prices[mask]
        same_side_sizes = recent_sizes[mask]

        # Price range normalized by average size
        price_range = np.max(same_side_prices) - np.min(same_side_prices)
        avg_size = np.mean(same_side_sizes)

        if avg_size < 1e-9:
            return 0.0

        # Low price impact despite large size = potential iceberg
        price_impact_ratio = price_range / (avg_size * 0.01 + 1e-9)

        return 1.0 / (1.0 + price_impact_ratio)  # Higher score = lower impact

    def analyze_tick(
        self,
        size: float,
        price: float,
        timestamp_ns: int,
        is_buyer_aggressor: bool,
        best_bid: float,
        best_ask: float,
    ) -> Optional[SniperSignal]:
        """
        Analyze incoming L3 tick for iceberg detection and generate sniper signal.
        """
        self._push_tick(size, price, timestamp_ns, is_buyer_aggressor)

        if not self.buffer_filled:
            return None

        # Detect anomalies
        size_anomaly, z_score = self._detect_size_anomaly()
        side = "buy" if is_buyer_aggressor else "sell"
        replenishment_rate = self._detect_replenishment(price, side)
        price_impact_score = self._calculate_price_impact(is_buyer_aggressor)

        # Calculate composite confidence score
        confidence_components = []

        if size_anomaly:
            confidence_components.append(min(abs(z_score) / 5.0, 1.0))

        if replenishment_rate > self.replenishment_rate_threshold:
            confidence_components.append(replenishment_rate)

        if price_impact_score > 0.5:
            confidence_components.append(price_impact_score)

        if not confidence_components:
            return None

        confidence = np.mean(confidence_components)

        # Estimate iceberg size using extreme value theory approximation
        tail_index = 1.0 / (abs(z_score) + 1e-9)
        estimated_iceberg = size * (1.0 + tail_index * confidence)

        if estimated_iceberg < self.min_iceberg_size:
            return None

        # Calculate urgency based on replenishment speed
        urgency_score = replenishment_rate * (1.0 + price_impact_score)

        # Determine signal direction
        signal_type = SignalType.BUY if is_buyer_aggressor else SignalType.SELL

        return SniperSignal(
            signal_type=signal_type,
            confidence=confidence,
            estimated_iceberg_size=estimated_iceberg,
            price_level=price,
            urgency_score=urgency_score,
        )

    def get_optimal_entry_price(
        self,
        signal: SniperSignal,
        current_mid: float,
        spread: float,
        risk_aversion: float = 0.5,
    ) -> float:
        """
        Calculate optimal entry price balancing fill probability vs adverse selection.
        Uses Kelly criterion-inspired sizing for price placement.
        """
        half_spread = spread / 2.0

        if signal.signal_type == SignalType.BUY:
            # Aggressive: hit the ask, Passive: provide at bid
            aggression_factor = signal.confidence * signal.urgency_score
            offset = half_spread * (1.0 - aggression_factor * (1.0 - risk_aversion))
            return current_mid - offset
        else:
            aggression_factor = signal.confidence * signal.urgency_score
            offset = half_spread * (1.0 - aggression_factor * (1.0 - risk_aversion))
            return current_mid + offset

    def reset(self) -> None:
        """Reset all internal state for new trading session."""
        self.trade_sizes.fill(0.0)
        self.trade_prices.fill(0.0)
        self.trade_timestamps.fill(0)
        self.aggressor_flags.fill(0)
        self.buffer_idx = 0
        self.buffer_filled = False
        self.mean_size = 0.0
        self.m2_size = 0.0
        self.count = 0
        self.last_price_levels.clear()
        self.replenishment_counts.clear()
