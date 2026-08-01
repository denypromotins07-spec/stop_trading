"""
Telemetry Module Root
Aggregates Python-side telemetry ensuring metric scraping never blocks GIL.
"""

from .prometheus_ml import PrometheusMLExporter, MLMetrics
from .alert_manager import AlertManager, AlertSeverity, AlertType

__all__ = [
    "PrometheusMLExporter",
    "MLMetrics",
    "AlertManager",
    "AlertSeverity",
    "AlertType",
]
