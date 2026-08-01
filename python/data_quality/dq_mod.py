"""
Data Quality Module Root.
Routes anomaly scores to Nautilus risk engine and Rust Global Kill Switch via ZMQ.
"""

import asyncio
from typing import Optional, Dict, Any, List
import logging
import zmq
import json
import time

from .isolation_forest import (
    IsolationForestDetector,
    AnomalyResult,
    get_isolation_forest_detector,
    initialize_detector_with_data,
    shutdown_dq_if_module,
)
from .autoencoder import (
    LightweightAutoencoder,
    ReconstructionResult,
    get_autoencoder,
    initialize_autoencoder_with_data,
    shutdown_dq_ae_module,
)

logger = logging.getLogger(__name__)


class DataQualityModule:
    """
    Central manager for data quality subsystem.
    Routes anomaly scores to risk engines and kill switch.
    """

    def __init__(
        self,
        zmq_endpoint: str = "tcp://localhost:5555",
        if_contamination: float = 0.05,
        ae_threshold: float = 0.5,
    ):
        self.zmq_endpoint = zmq_endpoint
        self.if_contamination = if_contamination
        self.ae_threshold = ae_threshold

        self._if_detector: Optional[IsolationForestDetector] = None
        self._autoencoder: Optional[LightweightAutoencoder] = None
        self._zmq_context: Optional[zmq.Context] = None
        self._zmq_socket: Optional[zmq.Socket] = None
        self._initialized = False

        self._anomaly_count = 0
        self._total_checks = 0
        self._kill_switch_triggered = False

    def initialize(self) -> bool:
        """Initialize all DQ components."""
        if self._initialized:
            return True

        try:
            # Initialize Isolation Forest
            self._if_detector = get_isolation_forest_detector(
                contamination=self.if_contamination,
            )

            # Initialize Autoencoder
            self._autoencoder = get_autoencoder(
                anomaly_threshold=self.ae_threshold,
            )

            # Initialize ZMQ socket
            self._zmq_context = zmq.Context()
            self._zmq_socket = self._zmq_context.socket(zmq.PUB)
            self._zmq_socket.connect(self.zmq_endpoint)

            self._initialized = True
            logger.info("Data Quality module initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize DQ module: {e}")
            return False

    def train_detectors(
        self,
        normal_data: List[np.ndarray],
        epochs: int = 100,
    ) -> Dict[str, Any]:
        """
        Train both detectors on normal data.

        Args:
            normal_data: List of normal feature vectors
            epochs: Training epochs for autoencoder

        Returns:
            Training results
        """
        import numpy as np
        X = np.vstack(normal_data) if isinstance(normal_data[0], np.ndarray) else np.array(normal_data)

        results = {}

        # Train Isolation Forest
        if self._if_detector:
            self._if_detector.fit(X)
            results["isolation_forest"] = self._if_detector.get_stats()

        # Train Autoencoder
        if self._autoencoder:
            ae_history = self._autoencoder.train(X, epochs=epochs)
            results["autoencoder"] = {
                **self._autoencoder.get_stats(),
                "training_history": ae_history,
            }

        logger.info(f"DQ detectors trained on {len(X)} samples")
        return results

    def check_quality(
        self,
        feature_vector: Any,
        ipc_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Check data quality using both detectors.

        Args:
            feature_vector: Feature vector from IPC
            ipc_payload: Optional full IPC payload

        Returns:
            Quality assessment dict
        """
        import numpy as np
        if not isinstance(feature_vector, np.ndarray):
            feature_vector = np.array(feature_vector)

        self._total_checks += 1

        # Run both detectors
        if_result = None
        ae_result = None

        if self._if_detector:
            if_result = self._if_detector.detect(feature_vector)

        if self._autoencoder:
            ae_result = self._autoencoder.detect(feature_vector)

        # Combine results
        is_anomaly = False
        anomaly_score = 0.0
        quarantine = False

        if if_result and if_result.is_anomaly:
            is_anomaly = True
            anomaly_score = max(anomaly_score, abs(if_result.anomaly_score))
            if if_result.quarantine_recommended:
                quarantine = True

        if ae_result and ae_result.is_anomaly:
            is_anomaly = True
            anomaly_score = max(anomaly_score, ae_result.reconstruction_error)
            if ae_result.anomaly_confidence > 0.8:
                quarantine = True

        if is_anomaly:
            self._anomaly_count += 1

        # Build result
        result = {
            "is_anomaly": is_anomaly,
            "anomaly_score": anomaly_score,
            "quarantine_recommended": quarantine,
            "if_result": {
                "is_anomaly": if_result.is_anomaly if if_result else False,
                "score": if_result.anomaly_score if if_result else 0.0,
            } if if_result else None,
            "ae_result": {
                "is_anomaly": ae_result.is_anomaly if ae_result else False,
                "error": ae_result.reconstruction_error if ae_result else 0.0,
            } if ae_result else None,
            "timestamp_ns": time.time_ns(),
        }

        # Route to risk engine
        self._route_to_risk_engine(result)

        # Check if kill switch should be triggered
        if anomaly_score > 0.9:
            self._trigger_kill_switch("critical_anomaly", anomaly_score)

        return result

    def _route_to_risk_engine(self, result: Dict[str, Any]):
        """Route anomaly score to Nautilus risk engine via ZMQ."""
        if not self._zmq_socket:
            return

        try:
            message = json.dumps({
                "type": "anomaly_score",
                "data": result,
            })
            self._zmq_socket.send_string(message)
        except Exception as e:
            logger.error(f"Failed to route to risk engine: {e}")

    def _trigger_kill_switch(self, reason: str, severity: float):
        """Trigger Rust Global Kill Switch via ZMQ."""
        if self._kill_switch_triggered:
            return  # Already triggered

        self._kill_switch_triggered = True

        kill_message = {
            "type": "kill_switch",
            "reason": reason,
            "severity": severity,
            "source": "data_quality",
            "timestamp_ns": time.time_ns(),
        }

        logger.critical(f"KILL SWITCH TRIGGERED: {reason} (severity: {severity})")

        try:
            if self._zmq_socket:
                message = json.dumps(kill_message)
                self._zmq_socket.send_string(message)
        except Exception as e:
            logger.error(f"Failed to send kill switch signal: {e}")

    def reset_kill_switch(self):
        """Reset the kill switch (requires manual authorization)."""
        self._kill_switch_triggered = False
        logger.info("Kill switch reset")

    def get_quarantined_payloads(
        self,
        payloads: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Filter and return payloads that should be quarantined.

        Args:
            payloads: List of IPC payloads

        Returns:
            List of quarantined payloads
        """
        quarantined = []

        for payload in payloads:
            features = payload.get("features")
            if features is None:
                continue

            result = self.check_quality(features, payload)
            if result["quarantine_recommended"]:
                payload["dq_result"] = result
                quarantined.append(payload)

        return quarantined

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive DQ statistics."""
        stats = {
            "initialized": self._initialized,
            "total_checks": self._total_checks,
            "anomaly_count": self._anomaly_count,
            "anomaly_rate": self._anomaly_count / max(1, self._total_checks),
            "kill_switch_triggered": self._kill_switch_triggered,
        }

        if self._if_detector:
            stats["isolation_forest"] = self._if_detector.get_stats()

        if self._autoencoder:
            stats["autoencoder"] = self._autoencoder.get_stats()

        return stats

    async def cleanup(self):
        """Cleanup resources."""
        await shutdown_dq_if_module()
        await shutdown_dq_ae_module()

        if self._zmq_socket:
            self._zmq_socket.close()
        if self._zmq_context:
            self._zmq_context.term()

        self._initialized = False
        logger.info("DQ module cleaned up")


# Module singleton
_module: Optional[DataQualityModule] = None


def get_dq_module(
    zmq_endpoint: str = "tcp://localhost:5555",
) -> DataQualityModule:
    """Get or create the DQ module singleton."""
    global _module
    if _module is None:
        _module = DataQualityModule(zmq_endpoint=zmq_endpoint)
        _module.initialize()
    return _module


async def initialize_dq(
    zmq_endpoint: str = "tcp://localhost:5555",
    training_data: Optional[List] = None,
) -> DataQualityModule:
    """Initialize DQ module with optional training."""
    module = get_dq_module(zmq_endpoint=zmq_endpoint)
    if not module._initialized:
        module.initialize()

    if training_data:
        module.train_detectors(training_data)

    return module


async def shutdown_dq_module():
    """Gracefully shutdown the DQ module."""
    global _module
    if _module:
        await _module.cleanup()
        _module = None
