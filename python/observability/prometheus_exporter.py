"""
Lightweight asynchronous FastAPI endpoint for Prometheus metrics.
Exposes Ray worker and Nautilus kernel metrics without blocking ML pipelines.
"""

from fastapi import FastAPI, Response
from typing import Dict, List, Optional, Any
import asyncio
import threading
import time
import numpy as np
from dataclasses import dataclass, field
from prometheus_client import (
    Counter, Gauge, Histogram, CollectorRegistry, generate_latest,
    CONTENT_TYPE_LATEST
)


# Prometheus metrics definitions
INFERENCE_LATENCY = Histogram(
    'ml_inference_latency_seconds',
    'ML inference latency in seconds',
    ['model_id', 'model_type']
)

QUEUE_DEPTH = Gauge(
    'inference_queue_depth',
    'Current queue depth for inference requests',
    ['worker_id']
)

MEMORY_USAGE = Gauge(
    'python_memory_usage_bytes',
    'Python process memory usage in bytes',
    ['component']
)

MODEL_ERRORS = Counter(
    'model_inference_errors_total',
    'Total number of model inference errors',
    ['model_id', 'error_type']
)

THROUGHPUT = Counter(
    'inference_throughput_total',
    'Total number of inferences processed',
    ['model_id']
)

REGIME_GAUGE = Gauge(
    'macro_regime_state',
    'Current macro regime state (0=RISK_ON, 1=RISK_OFF, 2=STAGFLATION)',
    []
)

ALPHA_SIGNAL = Gauge(
    'alpha_signal_value',
    'Current alpha signal value',
    ['strategy_id']
)

SHARPE_RATIO = Gauge(
    'model_sharpe_ratio',
    'Rolling Sharpe ratio for model',
    ['model_id']
)


@dataclass
class MetricsSnapshot:
    """Point-in-time metrics snapshot."""
    
    # Latency metrics
    avg_inference_latency_ms: float = 0.0
    p99_inference_latency_ms: float = 0.0
    
    # Queue metrics
    total_queue_depth: int = 0
    max_queue_depth: int = 0
    
    # Memory metrics
    python_memory_mb: float = 0.0
    ray_memory_mb: float = 0.0
    
    # Throughput
    inferences_per_second: float = 0.0
    
    # Model health
    active_models: int = 0
    quarantined_models: int = 0
    
    # Timestamp
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "avg_inference_latency_ms": self.avg_inference_latency_ms,
            "p99_inference_latency_ms": self.p99_inference_latency_ms,
            "total_queue_depth": self.total_queue_depth,
            "max_queue_depth": self.max_queue_depth,
            "python_memory_mb": self.python_memory_mb,
            "ray_memory_mb": self.ray_memory_mb,
            "inferences_per_second": self.inferences_per_second,
            "active_models": self.active_models,
            "quarantined_models": self.quarantined_models,
            "timestamp": self.timestamp
        }


class PrometheusExporter:
    """
    Asynchronous Prometheus metrics exporter.
    Thread-safe metric collection and exposure.
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        
        # Internal metrics storage
        self._latency_samples: Dict[str, List[float]] = {}
        self._queue_depths: Dict[str, int] = {}
        self._memory_usage: Dict[str, float] = {}
        self._throughput_counts: Dict[str, int] = {}
        self._error_counts: Dict[str, int] = {}
        
        # State tracking
        self._current_regime: int = 3  # Default TRANSITION
        self._alpha_signals: Dict[str, float] = {}
        self._sharpe_ratios: Dict[str, float] = {}
        
        # Timing
        self._start_time = time.time()
        self._last_throughput_reset = time.time()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # FastAPI app
        self.app = FastAPI(title="HFT Observability API")
        self._setup_routes()
    
    def _setup_routes(self) -> None:
        """Setup FastAPI routes."""
        @self.app.get("/metrics")
        async def get_metrics():
            """Prometheus metrics endpoint."""
            return Response(
                content=generate_latest(self.registry),
                media_type=CONTENT_TYPE_LATEST
            )
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "uptime_seconds": time.time() - self._start_time}
        
        @self.app.get("/snapshot")
        async def get_snapshot():
            """Get current metrics snapshot."""
            return self._get_snapshot().to_dict()
    
    def record_inference(
        self,
        model_id: str,
        model_type: str,
        latency_seconds: float,
        success: bool = True
    ) -> None:
        """Record an inference event."""
        with self._lock:
            # Record latency
            key = f"{model_id}_{model_type}"
            if key not in self._latency_samples:
                self._latency_samples[key] = []
            
            self._latency_samples[key].append(latency_seconds)
            
            # Keep last 1000 samples
            if len(self._latency_samples[key]) > 1000:
                self._latency_samples[key].pop(0)
            
            # Update Prometheus histogram
            INFERENCE_LATENCY.labels(model_id=model_id, model_type=model_type).observe(
                latency_seconds
            )
            
            # Record throughput
            if success:
                THROUGHPUT.labels(model_id=model_id).inc()
                
                if model_id not in self._throughput_counts:
                    self._throughput_counts[model_id] = 0
                self._throughput_counts[model_id] += 1
            else:
                MODEL_ERRORS.labels(model_id=model_id, error_type="inference_failed").inc()
    
    def update_queue_depth(self, worker_id: str, depth: int) -> None:
        """Update queue depth for a worker."""
        with self._lock:
            self._queue_depths[worker_id] = depth
            QUEUE_DEPTH.labels(worker_id=worker_id).set(depth)
    
    def update_memory_usage(
        self,
        component: str,
        memory_bytes: float
    ) -> None:
        """Update memory usage for a component."""
        with self._lock:
            self._memory_usage[component] = memory_bytes
            MEMORY_USAGE.labels(component=component).set(memory_bytes)
    
    def update_regime(self, regime: int) -> None:
        """Update current macro regime."""
        with self._lock:
            self._current_regime = regime
            REGIME_GAUGE.set(regime)
    
    def update_alpha_signal(self, strategy_id: str, value: float) -> None:
        """Update alpha signal for a strategy."""
        with self._lock:
            self._alpha_signals[strategy_id] = value
            ALPHA_SIGNAL.labels(strategy_id=strategy_id).set(value)
    
    def update_sharpe_ratio(self, model_id: str, sharpe: float) -> None:
        """Update Sharpe ratio for a model."""
        with self._lock:
            self._sharpe_ratios[model_id] = sharpe
            SHARPE_RATIO.labels(model_id=model_id).set(sharpe)
    
    def _get_snapshot(self) -> MetricsSnapshot:
        """Get current metrics snapshot."""
        with self._lock:
            snapshot = MetricsSnapshot()
            
            # Compute latency stats
            all_latencies = []
            for samples in self._latency_samples.values():
                all_latencies.extend(samples)
            
            if all_latencies:
                snapshot.avg_inference_latency_ms = np.mean(all_latencies) * 1000
                snapshot.p99_inference_latency_ms = np.percentile(all_latencies, 99) * 1000
            
            # Queue stats
            if self._queue_depths:
                snapshot.total_queue_depth = sum(self._queue_depths.values())
                snapshot.max_queue_depth = max(self._queue_depths.values())
            
            # Memory stats
            snapshot.python_memory_mb = self._memory_usage.get("python", 0) / (1024 * 1024)
            snapshot.ray_memory_mb = self._memory_usage.get("ray", 0) / (1024 * 1024)
            
            # Throughput
            elapsed = time.time() - self._last_throughput_reset
            if elapsed > 0:
                total_inferences = sum(self._throughput_counts.values())
                snapshot.inferences_per_second = total_inferences / elapsed
            
            # Model health
            snapshot.active_models = len(self._sharpe_ratios)
            # Quarantined would come from backtest module
            
            return snapshot
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application."""
        return self.app
    
    async def run_server(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Run the metrics server."""
        import uvicorn
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()


# Global singleton instance
_exporter_instance: Optional[PrometheusExporter] = None
_instance_lock = threading.Lock()


def get_prometheus_exporter() -> PrometheusExporter:
    """Get or create the global Prometheus exporter."""
    global _exporter_instance
    if _exporter_instance is None:
        with _instance_lock:
            if _exporter_instance is None:
                _exporter_instance = PrometheusExporter()
    return _exporter_instance


if __name__ == "__main__":
    # Test the exporter
    print("Testing PrometheusExporter:")
    
    exporter = PrometheusExporter()
    
    # Simulate some metrics
    for i in range(100):
        model_id = f"model_{i % 3}"
        latency = 0.001 + np.random.exponential(0.005)
        exporter.record_inference(model_id, "alpha", latency, success=np.random.random() > 0.05)
    
    exporter.update_queue_depth("worker_1", 5)
    exporter.update_queue_depth("worker_2", 3)
    
    exporter.update_memory_usage("python", 500 * 1024 * 1024)
    exporter.update_memory_usage("ray", 1000 * 1024 * 1024)
    
    exporter.update_regime(0)  # RISK_ON
    exporter.update_alpha_signal("crypto_momentum", 0.35)
    exporter.update_sharpe_ratio("model_0", 1.5)
    
    # Get snapshot
    snapshot = exporter._get_snapshot()
    print(f"\nMetrics Snapshot: {snapshot.to_dict()}")
    
    print("\nTo start server, run: uvicorn prometheus_exporter:app --host 0.0.0.0 --port 8000")
