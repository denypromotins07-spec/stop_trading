"""
Toxicity Module Root
Feeds toxicity probabilities to the pre-trade risk bus via the Rust IPC bridge.
Aggregates VPIN and informed flow signals for comprehensive toxicity assessment.
"""

import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import logging
import json

# Import toxicity components
try:
    from .vpin_ml import VPINForecaster, VPINFeatures, VPINPrediction
    from .informed_flow import InformedFlowClassifier, OrderBookShape, InformedFlowPrediction, FlowType
except ImportError:
    from vpin_ml import VPINForecaster, VPINFeatures, VPINPrediction
    from informed_flow import InformedFlowClassifier, OrderBookShape, InformedFlowPrediction, FlowType


logger = logging.getLogger(__name__)


@dataclass
class ToxicityReport:
    """Comprehensive toxicity assessment report."""
    timestamp_ns: int
    vpin_value: float
    vpin_level: str
    flow_type: str
    informed_probability: float
    adverse_selection_risk: float
    composite_toxicity_score: float
    recommended_spread_adjustment_bps: float
    confidence: float


class ToxicityModule:
    """
    Module root aggregating toxicity signals and feeding them to pre-trade risk.
    Communicates with Rust core via IPC bridge.
    """

    def __init__(
        self,
        ipc_topic: str = "toxicity.signals",
        update_interval_ms: int = 100,
        vpin_bucket_size: int = 100,
        min_institutional_size: float = 100.0,
    ):
        self.ipc_topic = ipc_topic
        self.update_interval_ms = update_interval_ms

        # Initialize forecasters
        self.vpin_forecaster = VPINForecaster(bucket_size=vpin_bucket_size)
        self.flow_classifier = InformedFlowClassifier(
            min_institutional_size=min_institutional_size
        )

        # State
        self._running = False
        self._ipc_bridge = None
        self._last_report: Optional[ToxicityReport] = None
        self._report_history: List[ToxicityReport] = []
        self._max_history = 1000

        # Metrics
        self.total_predictions = 0
        self.high_toxicity_count = 0

    async def start(self, ipc_bridge: Optional[Any] = None) -> None:
        """Start the toxicity module."""
        self._ipc_bridge = ipc_bridge
        self._running = True
        logger.info("ToxicityModule started")

        # Start update loop
        asyncio.create_task(self._update_loop())

    async def stop(self) -> None:
        """Stop the toxicity module."""
        self._running = False
        logger.info("ToxicityModule stopped")

    async def _update_loop(self) -> None:
        """Main update loop for toxicity calculations."""
        while self._running:
            try:
                await asyncio.sleep(self.update_interval_ms / 1000.0)
                await self._compute_and_publish()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in toxicity update loop: {e}")

    async def _compute_and_publish(self) -> None:
        """Compute toxicity metrics and publish to IPC."""
        # Get current VPIN prediction
        vpin_features = self._extract_vpin_features()
        vpin_prediction = self.vpin_forecaster.predict(vpin_features)

        # Get current flow classification
        ob_shape = self._get_current_order_book()
        if ob_shape:
            flow_prediction = self.flow_classifier.classify_flow(
                ob_shape=ob_shape,
                current_size=self._get_last_trade_size(),
                current_price=self._get_last_trade_price(),
                current_timestamp=self._get_current_timestamp(),
                is_buyer_aggressor=self._is_last_trade_buyer_aggressor(),
            )

            adverse_risk = self.flow_classifier.get_adverse_selection_risk(
                flow_prediction, "LONG"  # Default, should be strategy-specific
            )
        else:
            flow_prediction = InformedFlowPrediction(
                flow_type=FlowType.UNKNOWN,
                confidence=0.5,
                informed_probability=0.5,
                institutional_size_estimate=0.0,
                urgency_score=0.5,
            )
            adverse_risk = 0.5

        # Calculate composite toxicity score
        composite_score = self._calculate_composite_toxicity(
            vpin_prediction, flow_prediction, adverse_risk
        )

        # Calculate recommended spread adjustment
        base_spread = 1.0  # 1 bps base
        spread_adjustment = self.vpin_forecaster.get_spread_adjustment(
            vpin_prediction, base_spread * 100
        )

        # Create report
        report = ToxicityReport(
            timestamp_ns=self._get_current_timestamp(),
            vpin_value=vpin_prediction.vpin_value,
            vpin_level=vpin_prediction.toxicity_level,
            flow_type=flow_prediction.flow_type.value,
            informed_probability=flow_prediction.informed_probability,
            adverse_selection_risk=adverse_risk,
            composite_toxicity_score=composite_score,
            recommended_spread_adjustment_bps=spread_adjustment,
            confidence=min(vpin_prediction.confidence, flow_prediction.confidence),
        )

        self._last_report = report
        self._report_history.append(report)
        self.total_predictions += 1

        if composite_score > 0.7:
            self.high_toxicity_count += 1

        # Trim history
        if len(self._report_history) > self._max_history:
            self._report_history = self._report_history[-self._max_history:]

        # Publish to IPC
        await self._publish_to_ipc(report)

    def _calculate_composite_toxicity(
        self,
        vpin_pred: VPINPrediction,
        flow_pred: InformedFlowPrediction,
        adverse_risk: float,
    ) -> float:
        """Calculate composite toxicity score from multiple signals."""
        # Weight components
        vpin_weight = 0.4
        flow_weight = 0.3
        adverse_weight = 0.3

        # Normalize VPIN to 0-1 (already normalized)
        vpin_component = vpin_pred.vpin_value

        # Flow component based on informed probability
        flow_component = flow_pred.informed_probability

        # Adverse selection already 0-1
        adverse_component = adverse_risk

        composite = (
            vpin_weight * vpin_component +
            flow_weight * flow_component +
            adverse_weight * adverse_component
        )

        return float(np.clip(composite, 0.0, 1.0))

    def _extract_vpin_features(self) -> VPINFeatures:
        """Extract current VPIN features from market data."""
        # This would integrate with real market data feed
        # For now, return default features
        trades = self._get_recent_trades()

        if not trades:
            return VPINFeatures(
                buy_volume=0.0,
                sell_volume=0.0,
                trade_count=0,
                price_volatility=0.0,
                spread_bps=0.0,
                order_imbalance=0.0,
                trade_size_variance=0.0,
                aggressor_ratio=0.0,
                time_weighted_spread=0.0,
                volume_weighted_price=0.0,
            )

        prices = [t[1] for t in trades]
        spreads = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))] if len(prices) > 1 else [0.0]

        return self.vpin_forecaster.extract_features(trades, prices, spreads)

    def _get_current_order_book(self) -> Optional[OrderBookShape]:
        """Get current order book state."""
        # Placeholder - would integrate with real order book feed
        return None

    def _get_last_trade_size(self) -> float:
        """Get last trade size."""
        return 0.0

    def _get_last_trade_price(self) -> float:
        """Get last trade price."""
        return 0.0

    def _get_current_timestamp(self) -> int:
        """Get current timestamp in nanoseconds."""
        import time
        return time.time_ns()

    def _is_last_trade_buyer_aggressor(self) -> bool:
        """Check if last trade was buyer aggression."""
        return False

    def _get_recent_trades(self) -> List[tuple]:
        """Get recent trades for feature calculation."""
        return []

    async def _publish_to_ipc(self, report: ToxicityReport) -> None:
        """Publish toxicity report to Rust IPC bridge."""
        if not self._ipc_bridge:
            logger.debug("No IPC bridge available, skipping publish")
            return

        try:
            # Convert to dict for serialization
            report_dict = asdict(report)

            # Serialize to JSON for IPC
            message = json.dumps(report_dict)

            # Publish via IPC
            if hasattr(self._ipc_bridge, 'publish'):
                await self._ipc_bridge.publish(self.ipc_topic, message)
            elif hasattr(self._ipc_bridge, 'send'):
                await self._ipc_bridge.send(message)

            logger.debug(f"Published toxicity report: {report.composite_toxicity_score:.3f}")

        except Exception as e:
            logger.error(f"Failed to publish to IPC: {e}")

    def get_current_toxicity(self) -> Optional[ToxicityReport]:
        """Get the latest toxicity report."""
        return self._last_report

    def get_toxicity_metrics(self) -> Dict[str, Any]:
        """Get toxicity module metrics."""
        return {
            "total_predictions": self.total_predictions,
            "high_toxicity_count": self.high_toxicity_count,
            "high_toxicity_rate": self.high_toxicity_count / max(self.total_predictions, 1),
            "history_length": len(self._report_history),
        }

    def reset(self) -> None:
        """Reset all state."""
        self.vpin_forecaster.reset()
        self.flow_classifier.reset()
        self._report_history.clear()
        self._last_report = None
        self.total_predictions = 0
        self.high_toxicity_count = 0


# Import numpy at module level for composite calculation
import numpy as np


# Module-level singleton
_module_instance: Optional[ToxicityModule] = None


def get_module() -> ToxicityModule:
    """Get the module singleton instance."""
    global _module_instance
    if _module_instance is None:
        _module_instance = ToxicityModule()
    return _module_instance


async def initialize_module(ipc_bridge: Optional[Any] = None) -> ToxicityModule:
    """Initialize the toxicity module."""
    module = get_module()
    await module.start(ipc_bridge=ipc_bridge)
    return module


async def shutdown_module() -> None:
    """Shutdown the toxicity module."""
    module = get_module()
    await module.stop()
