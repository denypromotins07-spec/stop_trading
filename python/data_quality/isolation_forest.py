"""
Isolation Forest for Anomaly Detection.
Detects anomalous feature vectors from Rust IPC to quarantine toxic data.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    is_anomaly: bool
    anomaly_score: float
    confidence: float
    feature_vector: np.ndarray
    timestamp_ns: int
    quarantine_recommended: bool


class IsolationForestDetector:
    """
    Scikit-Learn Isolation Forest for detecting anomalous feature vectors.
    Quarantines toxic or malformed data from Rust IPC.
    """

    # Contamination rate (expected proportion of anomalies)
    CONTAMINATION = 0.05

    # Anomaly score threshold
    ANOMALY_THRESHOLD = -0.3

    # Quarantine threshold (more aggressive)
    QUARANTINE_THRESHOLD = -0.5

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = CONTAMINATION,
        max_samples: float = 'auto',
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.random_state = random_state

        self._model: Optional[IsolationForest] = None
        self._is_fitted = False
        self._training_samples: List[np.ndarray] = []
        self._detection_count = 0
        self._anomaly_count = 0

    def fit(
        self,
        X: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> 'IsolationForestDetector':
        """
        Fit the Isolation Forest model.

        Args:
            X: Training data of shape (n_samples, n_features)
            sample_weight: Optional sample weights

        Returns:
            Self
        """
        if len(X) < 10:
            logger.warning(f"Insufficient training samples: {len(X)}")
            return self

        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1,  # Use all CPU cores
        )

        self._model.fit(X, sample_weight=sample_weight)
        self._is_fitted = True
        self._training_samples = [X[i].copy() for i in range(min(1000, len(X)))]

        logger.info(f"Isolation Forest fitted with {len(X)} samples")
        return self

    def partial_fit(self, X: np.ndarray) -> 'IsolationForestDetector':
        """
        Incrementally update the model with new samples.
        Note: IsolationForest doesn't support true partial_fit, so we retrain periodically.
        """
        self._training_samples.extend([X[i].copy() for i in range(len(X))])

        # Keep only recent samples
        max_samples = 10000
        if len(self._training_samples) > max_samples:
            self._training_samples = self._training_samples[-max_samples:]

        # Retrain if we have enough new samples
        if len(self._training_samples) >= 100:
            X_train = np.vstack(self._training_samples)
            self.fit(X_train)

        return self

    def detect(
        self,
        feature_vector: np.ndarray,
    ) -> AnomalyResult:
        """
        Detect if a feature vector is anomalous.

        Args:
            feature_vector: Feature vector from IPC

        Returns:
            AnomalyResult
        """
        if not self._is_fitted:
            # If not fitted, assume normal
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                confidence=0.0,
                feature_vector=feature_vector,
                timestamp_ns=time.time_ns(),
                quarantine_recommended=False,
            )

        # Ensure correct shape
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)

        # Get anomaly score (negative = more anomalous)
        score = self._model.score_samples(feature_vector)[0]
        prediction = self._model.predict(feature_vector)[0]

        is_anomaly = prediction == -1
        quarantine = score < self.QUARANTINE_THRESHOLD

        self._detection_count += 1
        if is_anomaly:
            self._anomaly_count += 1

        # Calculate confidence based on distance from threshold
        if is_anomaly:
            confidence = min(1.0, abs(score - self.ANOMALY_THRESHOLD) / 0.5)
        else:
            confidence = min(1.0, abs(score - self.ANOMALY_THRESHOLD) / 0.5)

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=float(score),
            confidence=float(confidence),
            feature_vector=feature_vector.flatten(),
            timestamp_ns=time.time_ns(),
            quarantine_recommended=quarantine,
        )

    def detect_batch(
        self,
        feature_vectors: np.ndarray,
    ) -> List[AnomalyResult]:
        """
        Detect anomalies in a batch of vectors.

        Args:
            feature_vectors: Array of shape (n, d)

        Returns:
            List of AnomalyResult
        """
        if not self._is_fitted:
            return [
                AnomalyResult(
                    is_anomaly=False,
                    anomaly_score=0.0,
                    confidence=0.0,
                    feature_vector=v,
                    timestamp_ns=time.time_ns(),
                    quarantine_recommended=False,
                )
                for v in feature_vectors
            ]

        scores = self._model.score_samples(feature_vectors)
        predictions = self._model.predict(feature_vectors)

        results = []
        for i, (score, pred) in enumerate(zip(scores, predictions)):
            is_anomaly = pred == -1
            quarantine = score < self.QUARANTINE_THRESHOLD

            if is_anomaly:
                self._anomaly_count += 1

            results.append(AnomalyResult(
                is_anomaly=is_anomaly,
                anomaly_score=float(score),
                confidence=min(1.0, abs(score - self.ANOMALY_THRESHOLD) / 0.5),
                feature_vector=feature_vectors[i],
                timestamp_ns=time.time_ns(),
                quarantine_recommended=quarantine,
            ))

        self._detection_count += len(results)
        return results

    def get_quarantined_payloads(
        self,
        ipc_payloads: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Filter IPC payloads and return those that should be quarantined.

        Args:
            ipc_payloads: List of IPC payload dicts containing 'features'

        Returns:
            List of quarantined payloads
        """
        quarantined = []

        for payload in ipc_payloads:
            features = payload.get("features")
            if features is None:
                continue

            if not isinstance(features, np.ndarray):
                features = np.array(features)

            result = self.detect(features)
            if result.quarantine_recommended:
                payload["anomaly_score"] = result.anomaly_score
                payload["quarantine_reason"] = "isolation_forest"
                quarantined.append(payload)

        return quarantined

    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            "is_fitted": self._is_fitted,
            "n_estimators": self.n_estimators,
            "contamination": self.contamination,
            "training_samples": len(self._training_samples),
            "detection_count": self._detection_count,
            "anomaly_count": self._anomaly_count,
            "anomaly_rate": (
                self._anomaly_count / max(1, self._detection_count)
            ),
        }

    def reset(self):
        """Reset the detector."""
        self._model = None
        self._is_fitted = False
        self._training_samples = []
        self._detection_count = 0
        self._anomaly_count = 0
        logger.info("Isolation Forest detector reset")


# Module singleton
_detector: Optional[IsolationForestDetector] = None


def get_isolation_forest_detector(
    n_estimators: int = 100,
    contamination: float = 0.05,
) -> IsolationForestDetector:
    """Get or create the isolation forest detector singleton."""
    global _detector
    if _detector is None:
        _detector = IsolationForestDetector(
            n_estimators=n_estimators,
            contamination=contamination,
        )
    return _detector


def initialize_detector_with_data(
    training_data: np.ndarray,
    n_estimators: int = 100,
) -> IsolationForestDetector:
    """Initialize and fit the detector with training data."""
    global _detector
    _detector = IsolationForestDetector(n_estimators=n_estimators)
    _detector.fit(training_data)
    return _detector


async def shutdown_dq_if_module():
    """Shutdown the isolation forest module."""
    global _detector
    if _detector:
        _detector.reset()
        _detector = None
