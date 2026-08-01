"""
Informed Flow Classifier
Builds an informed flow classifier using order book shape and trade aggressor features.
Detects institutional vs retail flow patterns for adverse selection avoidance.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum
import logging


logger = logging.getLogger(__name__)


class FlowType(Enum):
    INSTITUTIONAL_BUY = "INSTITUTIONAL_BUY"
    INSTITUTIONAL_SELL = "INSTITUTIONAL_SELL"
    RETAIL_BUY = "RETAIL_BUY"
    RETAIL_SELL = "RETAIL_SELL"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"


@dataclass
class OrderBookShape:
    """Order book shape features."""
    bid_levels: List[Tuple[float, float]]  # (price, volume)
    ask_levels: List[Tuple[float, float]]
    bid_ask_spread: float
    mid_price: float
    total_bid_volume: float
    total_ask_volume: float
    weighted_bid_price: float
    weighted_ask_price: float


@dataclass
class TradeFlowFeatures:
    """Trade flow features for classification."""
    trade_sizes: List[float]
    aggressor_flags: List[bool]  # True = buyer aggressor
    trade_prices: List[float]
    trade_timestamps: List[int]  # nanoseconds


@dataclass
class InformedFlowPrediction:
    """Classification result for informed flow."""
    flow_type: FlowType
    confidence: float
    informed_probability: float
    institutional_size_estimate: float
    urgency_score: float


class InformedFlowClassifier:
    """
    Classifies order flow as informed (institutional) or uninformed (retail).
    Uses order book shape analysis and trade pattern recognition.
    """

    def __init__(
        self,
        min_institutional_size: float = 100.0,
        size_clustering_threshold: float = 0.2,
        timing_precision_ns: int = 50_000_000,  # 50ms
        lookback_trades: int = 50,
    ):
        self.min_institutional_size = min_institutional_size
        self.size_clustering_threshold = size_clustering_threshold
        self.timing_precision_ns = timing_precision_ns
        self.lookback_trades = lookback_trades

        # Rolling buffers
        self._trade_sizes: np.ndarray = np.zeros(lookback_trades, dtype=np.float64)
        self._aggressor_flags: np.ndarray = np.zeros(lookback_trades, dtype=bool)
        self._trade_prices: np.ndarray = np.zeros(lookback_trades, dtype=np.float64)
        self._trade_timestamps: np.ndarray = np.zeros(lookback_trades, dtype=np.int64)
        self._buffer_idx: int = 0
        self._filled_count: int = 0

        # Pre-allocated computation arrays
        self._size_diffs: np.ndarray = np.zeros(lookback_trades - 1, dtype=np.float64)
        self._time_diffs: np.ndarray = np.zeros(lookback_trades - 1, dtype=np.float64)

    def _push_trade(
        self,
        size: float,
        price: float,
        timestamp_ns: int,
        is_buyer_aggressor: bool,
    ) -> None:
        """Push trade into rolling buffer."""
        self._trade_sizes[self._buffer_idx] = size
        self._trade_prices[self._buffer_idx] = price
        self._trade_timestamps[self._buffer_idx] = timestamp_ns
        self._aggressor_flags[self._buffer_idx] = is_buyer_aggressor

        self._buffer_idx = (self._buffer_idx + 1) % self.lookback_trades
        if self._filled_count < self.lookback_trades:
            self._filled_count += 1

    def _calculate_size_clustering(self) -> float:
        """
        Calculate size clustering score.
        Institutional orders often show similar sizes (algorithmic splitting).
        Returns score from 0 (random) to 1 (highly clustered).
        """
        if self._filled_count < 5:
            return 0.0

        sizes = self._trade_sizes[: self._filled_count]

        # Calculate coefficient of variation
        mean_size = np.mean(sizes)
        std_size = np.std(sizes)

        if mean_size < 1e-9:
            return 0.0

        cv = std_size / mean_size

        # Lower CV = higher clustering
        # Transform to 0-1 scale where 1 = high clustering
        clustering_score = 1.0 / (1.0 + cv)

        return clustering_score

    def _calculate_timing_regularity(self) -> float:
        """
        Calculate timing regularity score.
        Institutional algorithms often trade at regular intervals.
        Returns score from 0 (random) to 1 (highly regular).
        """
        if self._filled_count < 3:
            return 0.0

        timestamps = self._trade_timestamps[: self._filled_count]
        time_diffs = np.diff(timestamps)

        if len(time_diffs) < 2:
            return 0.0

        # Calculate coefficient of variation in timing
        mean_diff = np.mean(time_diffs)
        std_diff = np.std(time_diffs)

        if mean_diff < 1e-9:
            return 0.0

        cv = std_diff / mean_diff

        # Lower CV = more regular timing
        regularity_score = 1.0 / (1.0 + cv)

        return regularity_score

    def _calculate_price_impact_efficiency(self) -> float:
        """
        Calculate price impact efficiency.
        Institutional traders minimize price impact per unit volume.
        Returns score from 0 (high impact) to 1 (low impact).
        """
        if self._filled_count < 5:
            return 0.0

        sizes = self._trade_sizes[: self._filled_count]
        prices = self._trade_prices[: self._filled_count]

        total_volume = np.sum(sizes)
        price_range = np.max(prices) - np.min(prices)
        mid_price = np.mean(prices)

        if mid_price < 1e-9 or total_volume < 1e-9:
            return 0.0

        # Price impact per unit volume (normalized)
        impact_per_volume = (price_range / mid_price) / total_volume

        # Transform to efficiency score (lower impact = higher score)
        efficiency = 1.0 / (1.0 + impact_per_volume * 1000)

        return efficiency

    def _analyze_order_book_imbalance(
        self,
        ob_shape: OrderBookShape,
    ) -> Tuple[float, float]:
        """
        Analyze order book imbalance for informed flow signals.
        Returns (imbalance_ratio, steepness_score).
        """
        if not ob_shape.bid_levels or not ob_shape.ask_levels:
            return 0.0, 0.0

        # Volume imbalance
        total_bid = ob_shape.total_bid_volume
        total_ask = ob_shape.total_ask_volume

        if total_bid + total_ask < 1e-9:
            imbalance = 0.0
        else:
            imbalance = (total_bid - total_ask) / (total_bid + total_ask)

        # Book steepness (concentration near best prices)
        if ob_shape.bid_levels:
            best_bid_vol = ob_shape.bid_levels[0][1]
            bid_steepness = best_bid_vol / (total_bid + 1e-9)
        else:
            bid_steepness = 0.0

        if ob_shape.ask_levels:
            best_ask_vol = ob_shape.ask_levels[0][1]
            ask_steepness = best_ask_vol / (total_ask + 1e-9)
        else:
            ask_steepness = 0.0

        steepness = (bid_steepness + ask_steepness) / 2.0

        return imbalance, steepness

    def _detect_sweeping_pattern(self) -> bool:
        """
        Detect aggressive sweeping pattern across price levels.
        Indicates urgent institutional demand/supply.
        """
        if self._filled_count < 3:
            return False

        prices = self._trade_prices[: self._filled_count]
        aggressors = self._aggressor_flags[: self._filled_count]
        timestamps = self._trade_timestamps[: self._filled_count]

        # Check for consecutive same-side aggression with price movement
        same_side_count = 0
        directional_moves = 0

        for i in range(1, self._filled_count):
            if aggressors[i] == aggressors[i - 1]:
                same_side_count += 1

                if aggressors[i]:  # Buyer aggression
                    if prices[i] >= prices[i - 1]:
                        directional_moves += 1
                else:  # Seller aggression
                    if prices[i] <= prices[i - 1]:
                        directional_moves += 1

        # Sweeping requires consistent same-side aggression with directional moves
        threshold = self._filled_count * 0.7
        return same_side_count >= threshold and directional_moves >= threshold * 0.8

    def classify_flow(
        self,
        ob_shape: OrderBookShape,
        current_size: float,
        current_price: float,
        current_timestamp: int,
        is_buyer_aggressor: bool,
    ) -> InformedFlowPrediction:
        """
        Classify current trade flow as informed or uninformed.
        """
        # Push current trade to buffer
        self._push_trade(current_size, current_price, current_timestamp, is_buyer_aggressor)

        # Need minimum history for reliable classification
        if self._filled_count < 10:
            return InformedFlowPrediction(
                flow_type=FlowType.UNKNOWN,
                confidence=0.3,
                informed_probability=0.5,
                institutional_size_estimate=0.0,
                urgency_score=0.5,
            )

        # Calculate feature scores
        size_clustering = self._calculate_size_clustering()
        timing_regularity = self._calculate_timing_regularity()
        impact_efficiency = self._calculate_price_impact_efficiency()
        ob_imbalance, ob_steepness = self._analyze_order_book_imbalance(ob_shape)
        is_sweeping = self._detect_sweeping_pattern()

        # Composite informed probability
        informed_components = [
            size_clustering * 0.25,
            timing_regularity * 0.20,
            impact_efficiency * 0.20,
            (1.0 - abs(ob_imbalance)) * 0.15,  # Extreme imbalance can be retail FOMO
            ob_steepness * 0.10,
            (1.0 if is_sweeping else 0.3) * 0.10,
        ]

        informed_probability = sum(informed_components)
        informed_probability = np.clip(informed_probability, 0.0, 1.0)

        # Determine flow type
        avg_size = np.mean(self._trade_sizes[: self._filled_count])
        buy_aggression = np.mean(self._aggressor_flags[: self._filled_count])

        if informed_probability > 0.6:
            if avg_size >= self.min_institutional_size:
                if buy_aggression > 0.6:
                    flow_type = FlowType.INSTITUTIONAL_BUY
                elif buy_aggression < 0.4:
                    flow_type = FlowType.INSTITUTIONAL_SELL
                else:
                    flow_type = FlowType.MIXED
            else:
                flow_type = FlowType.MIXED
        else:
            if buy_aggression > 0.6:
                flow_type = FlowType.RETAIL_BUY
            elif buy_aggression < 0.4:
                flow_type = FlowType.RETAIL_SELL
            else:
                flow_type = FlowType.MIXED

        # Confidence based on sample size and feature consistency
        base_confidence = min(self._filled_count / self.lookback_trades, 1.0)
        feature_variance = np.var(informed_components)
        confidence = base_confidence * (1.0 - feature_variance * 2)
        confidence = np.clip(confidence, 0.1, 0.95)

        # Estimate institutional size
        if flow_type in [FlowType.INSTITUTIONAL_BUY, FlowType.INSTITUTIONAL_SELL]:
            institutional_estimate = avg_size * (1.0 + informed_probability)
        else:
            institutional_estimate = avg_size * informed_probability

        # Urgency score
        urgency = 0.5
        if is_sweeping:
            urgency = 0.9
        elif timing_regularity > 0.7:
            urgency = 0.7
        elif current_size > avg_size * 2:
            urgency = 0.8

        return InformedFlowPrediction(
            flow_type=flow_type,
            confidence=float(confidence),
            informed_probability=float(informed_probability),
            institutional_size_estimate=float(institutional_estimate),
            urgency_score=float(urgency),
        )

    def get_adverse_selection_risk(
        self,
        prediction: InformedFlowPrediction,
        position_side: str,  # "LONG" or "SHORT"
    ) -> float:
        """
        Calculate adverse selection risk given current position.
        Higher risk when trading against informed flow.
        """
        if prediction.flow_type == FlowType.UNKNOWN:
            return 0.5

        # Risk matrix
        risk_map = {
            ("LONG", FlowType.INSTITUTIONAL_SELL): 0.9,
            ("LONG", FlowType.INSTITUTIONAL_BUY): 0.2,
            ("SHORT", FlowType.INSTITUTIONAL_BUY): 0.9,
            ("SHORT", FlowType.INSTITUTIONAL_SELL): 0.2,
            ("LONG", FlowType.RETAIL_SELL): 0.4,
            ("LONG", FlowType.RETAIL_BUY): 0.3,
            ("SHORT", FlowType.RETAIL_BUY): 0.4,
            ("SHORT", FlowType.RETAIL_SELL): 0.3,
        }

        base_risk = risk_map.get((position_side, prediction.flow_type), 0.5)

        # Adjust by informed probability
        adjusted_risk = base_risk * prediction.informed_probability
        adjusted_risk += 0.5 * (1.0 - prediction.informed_probability)

        return float(np.clip(adjusted_risk, 0.0, 1.0))

    def reset(self) -> None:
        """Reset all internal state."""
        self._trade_sizes.fill(0.0)
        self._aggressor_flags.fill(False)
        self._trade_prices.fill(0.0)
        self._trade_timestamps.fill(0)
        self._buffer_idx = 0
        self._filled_count = 0
