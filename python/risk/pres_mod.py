"""
Risk Module Root
Aggregates ML risk metrics and pushes them to global risk management actors.
Integrates drawdown prediction, capital preservation, and toxicity signals.
"""

import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import logging
import json

# Import risk components
try:
    from .drawdown_seq import DrawdownPredictorLSTM, DrawdownPrediction
    from .capital_preserver import CapitalPreservationEngine, CapitalPreservationState, StressLevel
except ImportError:
    from drawdown_seq import DrawdownPredictorLSTM, DrawdownPrediction
    from capital_preserver import CapitalPreservationEngine, CapitalPreservationState, StressLevel


logger = logging.getLogger(__name__)


@dataclass
class RiskMetricsReport:
    """Comprehensive risk metrics report."""
    timestamp_ns: int
    current_capital: float
    peak_capital: float
    current_drawdown: float
    predicted_drawdown: float
    breach_probability: float
    stress_level: str
    position_scale_factor: float
    recommended_leverage: float
    should_halt_trading: bool
    risk_flags: List[str]
    composite_risk_score: float


class RiskModule:
    """
    Module root aggregating all ML-based risk metrics.
    Pushes unified risk signals to global risk management via IPC.
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        ipc_topic: str = "risk.metrics",
        update_interval_ms: int = 100,
        max_drawdown_limit: float = 0.05,
        lstm_sequence_length: int = 50,
        lstm_feature_dim: int = 8,
        delever_threshold: float = 0.5,
    ):
        """
        Initialize the risk module.

        Args:
            initial_capital: Starting capital for preservation engine
            ipc_topic: Topic for publishing risk metrics
            update_interval_ms: Update frequency in milliseconds
            max_drawdown_limit: Maximum allowed drawdown
            lstm_sequence_length: Sequence length for LSTM predictor
            lstm_feature_dim: Feature dimension for LSTM
            delever_threshold: Breach probability threshold for deleveraging
        """
        self.initial_capital = initial_capital
        self.ipc_topic = ipc_topic
        self.update_interval_ms = update_interval_ms
        self.max_drawdown_limit = max_drawdown_limit
        self.delever_threshold = delever_threshold

        # Initialize components
        self.drawdown_predictor = DrawdownPredictorLSTM(
            sequence_length=lstm_sequence_length,
            feature_dim=lstm_feature_dim,
            max_drawdown_threshold=max_drawdown_limit,
        )
        self.capital_preserver = CapitalPreservationEngine(
            initial_capital=initial_capital,
            max_drawdown_limit=max_drawdown_limit,
        )

        # State
        self._running = False
        self._ipc_bridge = None
        self._last_report: Optional[RiskMetricsReport] = None
        self._report_history: List[RiskMetricsReport] = []
        self._max_history = 500

        # Metrics
        self.total_updates = 0
        self.delever_triggers = 0
        self.halt_events = 0

    async def start(self, ipc_bridge: Optional[Any] = None) -> None:
        """Start the risk module."""
        self._ipc_bridge = ipc_bridge
        self._running = True
        logger.info("RiskModule started")

        # Start update loop
        asyncio.create_task(self._update_loop())

    async def stop(self) -> None:
        """Stop the risk module."""
        self._running = False
        logger.info("RiskModule stopped")

    async def _update_loop(self) -> None:
        """Main update loop for risk calculations."""
        while self._running:
            try:
                await asyncio.sleep(self.update_interval_ms / 1000.0)
                await self._compute_and_publish()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in risk update loop: {e}")

    async def _compute_and_publish(self) -> None:
        """Compute risk metrics and publish to IPC."""
        # Get current capital state
        current_capital = self._get_current_capital()
        cap_state = self.capital_preserver.update_capital(current_capital)

        # Get drawdown prediction
        dd_prediction = self._get_drawdown_prediction()

        # Check if deleveraging is needed
        should_delever = False
        if dd_prediction and dd_prediction.breach_probability > self.delever_threshold:
            should_delever = True
            self.delever_triggers += 1

        # Check if trading should halt
        should_halt = self.capital_preserver.should_halt_trading()
        if should_halt:
            self.halt_events += 1

        # Calculate composite risk score
        composite_risk = self._calculate_composite_risk(
            cap_state, dd_prediction
        )

        # Build report
        report = RiskMetricsReport(
            timestamp_ns=self._get_current_timestamp(),
            current_capital=cap_state.current_capital,
            peak_capital=cap_state.peak_capital,
            current_drawdown=cap_state.current_drawdown,
            predicted_drawdown=dd_prediction.predicted_drawdown if dd_prediction else 0.0,
            breach_probability=dd_prediction.breach_probability if dd_prediction else 0.0,
            stress_level=cap_state.stress_level.value,
            position_scale_factor=cap_state.position_scale_factor,
            recommended_leverage=dd_prediction.recommended_leverage if dd_prediction else 1.0,
            should_halt_trading=should_halt,
            risk_flags=self._get_active_risk_flags(cap_state, dd_prediction),
            composite_risk_score=composite_risk,
        )

        self._last_report = report
        self._report_history.append(report)
        self.total_updates += 1

        # Trim history
        if len(self._report_history) > self._max_history:
            self._report_history = self._report_history[-self._max_history:]

        # Publish to IPC
        await self._publish_to_ipc(report)

        # Log critical events
        if should_halt:
            logger.critical("HALT TRADING triggered by risk module")
        elif should_delever:
            logger.warning(f"Deleveraging triggered: breach_prob={dd_prediction.breach_probability:.3f}")

    def _get_current_capital(self) -> float:
        """Get current capital value (placeholder - integrates with portfolio system)."""
        # In production, this would query the actual portfolio value
        return self.capital_preserver._current_capital

    def _get_drawdown_prediction(self) -> Optional[DrawdownPrediction]:
        """Get drawdown prediction from LSTM model."""
        # Add synthetic features for demonstration
        # In production, these would come from the Rust feature store
        features = self._generate_risk_features()
        if features is not None:
            self.drawdown_predictor.add_feature_vector(features)
        return self.drawdown_predictor.predict()

    def _generate_risk_features(self) -> Optional[np.ndarray]:
        """Generate risk features for LSTM input."""
        # Placeholder features - in production these come from Rust feature store
        # Features might include: returns, volatility, volume, spread, etc.
        try:
            import numpy as np

            # Generate synthetic features based on current state
            current_dd = (
                self.capital_preserver._peak_capital - 
                self.capital_preserver._current_capital
            ) / self.capital_preserver._peak_capital

            # 8 features matching LSTM config
            features = np.array([
                current_dd,  # Current drawdown
                current_dd * 10,  # Scaled drawdown
                0.01,  # Recent volatility (placeholder)
                0.0,  # Returns momentum (placeholder)
                1.0,  # Volume ratio (placeholder)
                0.5,  # Spread level (placeholder)
                0.3,  # Toxicity score (placeholder)
                0.1,  # External risk signal (placeholder)
            ], dtype=np.float32)

            return features
        except ImportError:
            return None

    def _calculate_composite_risk(
        self,
        cap_state: CapitalPreservationState,
        dd_prediction: Optional[DrawdownPrediction],
    ) -> float:
        """Calculate composite risk score from all signals."""
        components = []

        # Current drawdown component (0-1 scale)
        dd_component = min(cap_state.current_drawdown / self.max_drawdown_limit, 1.0)
        components.append(dd_component * 0.3)

        # Predicted breach probability
        if dd_prediction:
            components.append(dd_prediction.breach_probability * 0.4)
        else:
            components.append(0.0)

        # Stress level component
        stress_scores = {
            "NORMAL": 0.0,
            "ELEVATED": 0.3,
            "HIGH": 0.6,
            "CRITICAL": 1.0,
        }
        components.append(stress_scores.get(cap_state.stress_level.value, 0.0) * 0.3)

        return float(np.clip(sum(components), 0.0, 1.0))

    def _get_active_risk_flags(
        self,
        cap_state: CapitalPreservationState,
        dd_prediction: Optional[DrawdownPrediction],
    ) -> List[str]:
        """Get list of active risk flags."""
        flags = []

        # Capital preservation flags
        if cap_state.current_drawdown > self.max_drawdown_limit * 0.8:
            flags.append("DRAWDOWN_WARNING")
        if cap_state.position_scale_factor < 0.5:
            flags.append("POSITION_REDUCED")

        # Prediction flags
        if dd_prediction:
            if dd_prediction.risk_level in ["HIGH", "CRITICAL"]:
                flags.append(f"PREDICTED_{dd_prediction.risk_level}")
            if dd_prediction.breach_probability > self.delever_threshold:
                flags.append("DELEVER_SIGNAL")

        # Add flags from capital preserver
        flags.extend(cap_state.__dict__.get('risk_flags', []))

        return flags

    def _get_current_timestamp(self) -> int:
        """Get current timestamp in nanoseconds."""
        import time
        return time.time_ns()

    async def _publish_to_ipc(self, report: RiskMetricsReport) -> None:
        """Publish risk report to Rust IPC bridge."""
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

            logger.debug(f"Published risk report: score={report.composite_risk_score:.3f}")

        except Exception as e:
            logger.error(f"Failed to publish to IPC: {e}")

    def get_current_risk(self) -> Optional[RiskMetricsReport]:
        """Get the latest risk report."""
        return self._last_report

    def should_delever(self) -> bool:
        """Check if deleveraging is currently recommended."""
        if self._last_report:
            return self._last_report.breach_probability > self.delever_threshold
        return False

    def should_halt(self) -> bool:
        """Check if trading should be halted."""
        if self._last_report:
            return self._last_report.should_halt_trading
        return False

    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get risk module metrics."""
        return {
            "total_updates": self.total_updates,
            "delever_triggers": self.delever_triggers,
            "halt_events": self.halt_events,
            "history_length": len(self._report_history),
            "current_composite_risk": self._last_report.composite_risk_score if self._last_report else 0.0,
        }

    def reset(self) -> None:
        """Reset all state."""
        self.drawdown_predictor.reset()
        self.capital_preserver.reset(self.initial_capital)
        self._report_history.clear()
        self._last_report = None
        self.total_updates = 0
        self.delever_triggers = 0
        self.halt_events = 0


# Import numpy at module level
import numpy as np


# Module-level singleton
_module_instance: Optional[RiskModule] = None


def get_module() -> RiskModule:
    """Get the module singleton instance."""
    global _module_instance
    if _module_instance is None:
        _module_instance = RiskModule()
    return _module_instance


async def initialize_module(
    initial_capital: float = 1_000_000.0,
    ipc_bridge: Optional[Any] = None,
) -> RiskModule:
    """Initialize the risk module."""
    module = RiskModule(initial_capital=initial_capital)
    await module.start(ipc_bridge=ipc_bridge)
    return module


async def shutdown_module() -> None:
    """Shutdown the risk module."""
    module = get_module()
    await module.stop()
