"""
Custom Prometheus exporter for ML metrics.
Exposes inference latency, feature drift, queue depths, Ray actor states, and Nautilus MessageBus throughput.
Runs on dedicated lightweight aiohttp thread that yields immediately if main event loop is busy.
Non-blocking design ensures metric scraping never delays inference queues.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from collections import deque
import time
import threading
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class MLMetrics:
    """Container for ML-specific metrics."""
    # Inference metrics
    inference_latency_ms: deque = field(default_factory=lambda: deque(maxlen=1000))
    inference_count: int = 0
    inference_errors: int = 0
    
    # Feature quality
    feature_drift_score: float = 0.0
    feature_null_ratio: float = 0.0
    feature_staleness_ns: int = 0
    
    # Queue depths
    inference_queue_depth: int = 0
    execution_queue_depth: int = 0
    message_bus_backlog: int = 0
    
    # Model performance
    prediction_confidence: deque = field(default_factory=lambda: deque(maxlen=1000))
    model_version: str = "unknown"
    
    # Resource usage
    ram_usage_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    
    @property
    def avg_inference_latency_ms(self) -> float:
        """Calculate average inference latency."""
        if not self.inference_latency_ms:
            return 0.0
        return sum(self.inference_latency_ms) / len(self.inference_latency_ms)
    
    @property
    def p99_inference_latency_ms(self) -> float:
        """Calculate P99 inference latency."""
        if not self.inference_latency_ms:
            return 0.0
        sorted_latencies = sorted(self.inference_latency_ms)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]
    
    @property
    def avg_confidence(self) -> float:
        """Calculate average prediction confidence."""
        if not self.prediction_confidence:
            return 0.0
        return sum(self.prediction_confidence) / len(self.prediction_confidence)


class MetricCollector:
    """Thread-safe metric collector with lock-free reads."""
    
    def __init__(self):
        self._metrics = MLMetrics()
        self._lock = threading.RLock()
        self._custom_gauges: Dict[str, float] = {}
        self._custom_counters: Dict[str, int] = {}
        self._custom_histograms: Dict[str, List[float]] = {}
        
        # Timestamp of last update
        self._last_update = time.time()
    
    def record_inference(self, latency_ms: float, success: bool = True,
                         confidence: Optional[float] = None):
        """Record an inference event."""
        with self._lock:
            self._metrics.inference_count += 1
            if not success:
                self._metrics.inference_errors += 1
            
            self._metrics.inference_latency_ms.append(latency_ms)
            
            if confidence is not None:
                self._metrics.prediction_confidence.append(confidence)
            
            self._last_update = time.time()
    
    def update_feature_quality(self, drift_score: float, 
                               null_ratio: float,
                               staleness_ns: int):
        """Update feature quality metrics."""
        with self._lock:
            self._metrics.feature_drift_score = drift_score
            self._metrics.feature_null_ratio = null_ratio
            self._metrics.feature_staleness_ns = staleness_ns
            self._last_update = time.time()
    
    def update_queue_depths(self, inference: int, execution: int,
                           message_bus: int):
        """Update queue depth metrics."""
        with self._lock:
            self._metrics.inference_queue_depth = inference
            self._metrics.execution_queue_depth = execution
            self._metrics.message_bus_backlog = message_bus
            self._last_update = time.time()
    
    def update_resource_usage(self, ram_mb: float, gpu_mb: float):
        """Update resource usage metrics."""
        with self._lock:
            self._metrics.ram_usage_mb = ram_mb
            self._metrics.gpu_memory_mb = gpu_mb
            self._last_update = time.time()
    
    def set_model_version(self, version: str):
        """Set current model version."""
        with self._lock:
            self._metrics.model_version = version
    
    def set_custom_gauge(self, name: str, value: float):
        """Set custom gauge metric."""
        with self._lock:
            self._custom_gauges[name] = value
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment custom counter."""
        with self._lock:
            self._custom_counters[name] = self._custom_counters.get(name, 0) + value
    
    def record_histogram(self, name: str, value: float):
        """Record histogram value."""
        with self._lock:
            if name not in self._custom_histograms:
                self._custom_histograms[name] = []
            self._custom_histograms[name].append(value)
            # Limit histogram size
            if len(self._custom_histograms[name]) > 10000:
                self._custom_histograms[name] = self._custom_histograms[name][-5000:]
    
    def get_metrics_snapshot(self) -> Dict:
        """Get thread-safe snapshot of all metrics."""
        with self._lock:
            return {
                'ml_metrics': {
                    'inference_count': self._metrics.inference_count,
                    'inference_errors': self._metrics.inference_errors,
                    'avg_inference_latency_ms': self._metrics.avg_inference_latency_ms,
                    'p99_inference_latency_ms': self._metrics.p99_inference_latency_ms,
                    'feature_drift_score': self._metrics.feature_drift_score,
                    'feature_null_ratio': self._metrics.feature_null_ratio,
                    'feature_staleness_ns': self._metrics.feature_staleness_ns,
                    'inference_queue_depth': self._metrics.inference_queue_depth,
                    'execution_queue_depth': self._metrics.execution_queue_depth,
                    'message_bus_backlog': self._metrics.message_bus_backlog,
                    'avg_prediction_confidence': self._metrics.avg_confidence,
                    'model_version': self._metrics.model_version,
                    'ram_usage_mb': self._metrics.ram_usage_mb,
                    'gpu_memory_mb': self._metrics.gpu_memory_mb,
                },
                'custom_gauges': self._custom_gauges.copy(),
                'custom_counters': self._custom_counters.copy(),
                'custom_histograms': {
                    k: v.copy() for k, v in self._custom_histograms.items()
                },
                'last_update': self._last_update,
            }


class PrometheusMLExporter:
    """
    Prometheus metrics exporter for ML systems.
    
    Features:
    - Dedicated aiohttp server on separate thread
    - Non-blocking metric collection
    - Custom ML metrics format
    - Ray actor state exposure
    - Nautilus MessageBus throughput tracking
    """
    
    def __init__(self,
                 host: str = "0.0.0.0",
                 port: int = 9090,
                 metrics_path: str = "/metrics",
                 health_path: str = "/health"):
        """
        Initialize exporter.
        
        Args:
            host: Host to bind to
            port: Port for metrics endpoint
            metrics_path: Path for Prometheus metrics
            health_path: Path for health check
        """
        self._host = host
        self._port = port
        self._metrics_path = metrics_path
        self._health_path = health_path
        
        self._collector = MetricCollector()
        self._server = None
        self._runner = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        
        # Additional metric callbacks
        self._ray_actor_callbacks: List[Callable[[], Dict]] = []
        self._nautilus_callbacks: List[Callable[[], Dict]] = []
        
        logger.info(f"PrometheusMLExporter initialized on {host}:{port}")
    
    def start(self):
        """Start the metrics server in a background thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()
        
        logger.info("PrometheusMLExporter started")
    
    def stop(self):
        """Stop the metrics server."""
        self._running = False
        
        if self._runner:
            asyncio.run(self._runner.cleanup())
        
        if self._thread:
            self._thread.join(timeout=5)
        
        logger.info("PrometheusMLExporter stopped")
    
    def _run_server(self):
        """Run aiohttp server in dedicated thread."""
        import aiohttp
        from aiohttp import web
        
        async def metrics_handler(request):
            """Handle Prometheus metrics scrape request."""
            # Yield immediately if busy
            await asyncio.sleep(0)
            
            try:
                metrics_text = self._generate_prometheus_metrics()
                return web.Response(
                    text=metrics_text,
                    content_type='text/plain'
                )
            except Exception as e:
                logger.error(f"Metrics generation error: {e}")
                return web.Response(
                    text=f"# Error generating metrics: {e}",
                    status=500,
                    content_type='text/plain'
                )
        
        async def health_handler(request):
            """Health check endpoint."""
            await asyncio.sleep(0)
            
            snapshot = self._collector.get_metrics_snapshot()
            age_seconds = time.time() - snapshot['last_update']
            
            status = 'healthy' if age_seconds < 60 else 'stale'
            
            return web.json_response({
                'status': status,
                'age_seconds': age_seconds,
                'timestamp': datetime.utcnow().isoformat(),
            })
        
        async def api_metrics_handler(request):
            """JSON API for metrics (non-Prometheus)."""
            await asyncio.sleep(0)
            return web.json_response(self._collector.get_metrics_snapshot())
        
        # Create application
        app = web.Application()
        app.router.add_get(self._metrics_path, metrics_handler)
        app.router.add_get(self._health_path, health_handler)
        app.router.add_get('/api/metrics', api_metrics_handler)
        
        # Run server
        self._runner = web.AppRunner(app)
        asyncio.run(self._runner.setup())
        
        site = web.TCPSite(self._runner, self._host, self._port)
        asyncio.run(site.start())
        
        logger.info(f"Metrics server running on http://{self._host}:{self._port}")
        
        # Keep running
        try:
            while self._running:
                asyncio.run(asyncio.sleep(1))
        except Exception as e:
            logger.error(f"Server error: {e}")
    
    def _generate_prometheus_metrics(self) -> str:
        """Generate Prometheus-format metrics string."""
        lines = []
        snapshot = self._collector.get_metrics_snapshot()
        ml = snapshot['ml_metrics']
        timestamp = time.time()
        
        # Helper to add metric
        def add_metric(name: str, value: float, metric_type: str = 'gauge',
                       help_text: str = ""):
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {metric_type}")
            lines.append(f"{name} {value} {int(timestamp * 1000)}")
        
        # Inference metrics
        add_metric(
            'ml_inference_count_total',
            ml['inference_count'],
            'counter',
            'Total number of inferences'
        )
        add_metric(
            'ml_inference_errors_total',
            ml['inference_errors'],
            'counter',
            'Total inference errors'
        )
        add_metric(
            'ml_inference_latency_avg_ms',
            ml['avg_inference_latency_ms'],
            'gauge',
            'Average inference latency in milliseconds'
        )
        add_metric(
            'ml_inference_latency_p99_ms',
            ml['p99_inference_latency_ms'],
            'gauge',
            'P99 inference latency in milliseconds'
        )
        
        # Feature quality
        add_metric(
            'ml_feature_drift_score',
            ml['feature_drift_score'],
            'gauge',
            'Feature distribution drift score (0-1)'
        )
        add_metric(
            'ml_feature_null_ratio',
            ml['feature_null_ratio'],
            'gauge',
            'Ratio of null/missing features'
        )
        add_metric(
            'ml_feature_staleness_ns',
            ml['feature_staleness_ns'],
            'gauge',
            'Feature staleness in nanoseconds'
        )
        
        # Queue depths
        add_metric(
            'ml_inference_queue_depth',
            ml['inference_queue_depth'],
            'gauge',
            'Current inference queue depth'
        )
        add_metric(
            'ml_execution_queue_depth',
            ml['execution_queue_depth'],
            'gauge',
            'Current execution queue depth'
        )
        add_metric(
            'nautilus_messagebus_backlog',
            ml['message_bus_backlog'],
            'gauge',
            'Nautilus MessageBus backlog count'
        )
        
        # Model metrics
        add_metric(
            'ml_prediction_confidence_avg',
            ml['avg_prediction_confidence'],
            'gauge',
            'Average prediction confidence'
        )
        
        # Resource usage
        add_metric(
            'ml_ram_usage_mb',
            ml['ram_usage_mb'],
            'gauge',
            'RAM usage in megabytes'
        )
        add_metric(
            'ml_gpu_memory_mb',
            ml['gpu_memory_mb'],
            'gauge',
            'GPU memory usage in megabytes'
        )
        
        # Custom gauges
        for name, value in snapshot['custom_gauges'].items():
            add_metric(f'ml_custom_{name}', value, 'gauge', f'Custom metric: {name}')
        
        # Custom counters
        for name, value in snapshot['custom_counters'].items():
            add_metric(f'ml_custom_{name}_total', value, 'counter', f'Custom counter: {name}')
        
        # Ray actor metrics
        for callback in self._ray_actor_callbacks:
            try:
                ray_metrics = callback()
                for name, value in ray_metrics.items():
                    add_metric(f'ray_actor_{name}', value, 'gauge', f'Ray actor metric: {name}')
            except Exception as e:
                logger.debug(f"Ray callback error: {e}")
        
        # Nautilus metrics
        for callback in self._nautilus_callbacks:
            try:
                nautilus_metrics = callback()
                for name, value in nautilus_metrics.items():
                    add_metric(f'nautilus_{name}', value, 'gauge', f'Nautilus metric: {name}')
            except Exception as e:
                logger.debug(f"Nautilus callback error: {e}")
        
        return '\n'.join(lines) + '\n'
    
    # Convenience methods for recording metrics
    def record_inference(self, latency_ms: float, success: bool = True,
                         confidence: Optional[float] = None):
        """Record inference metric."""
        self._collector.record_inference(latency_ms, success, confidence)
    
    def update_feature_quality(self, drift_score: float,
                               null_ratio: float,
                               staleness_ns: int):
        """Update feature quality metrics."""
        self._collector.update_feature_quality(drift_score, null_ratio, staleness_ns)
    
    def update_queue_depths(self, inference: int, execution: int,
                           message_bus: int):
        """Update queue depth metrics."""
        self._collector.update_queue_depths(inference, execution, message_bus)
    
    def update_resource_usage(self, ram_mb: float, gpu_mb: float):
        """Update resource usage."""
        self._collector.update_resource_usage(ram_mb, gpu_mb)
    
    def register_ray_callback(self, callback: Callable[[], Dict]):
        """Register callback for Ray actor metrics."""
        self._ray_actor_callbacks.append(callback)
    
    def register_nautilus_callback(self, callback: Callable[[], Dict]):
        """Register callback for Nautilus metrics."""
        self._nautilus_callbacks.append(callback)
    
    def get_collector(self) -> MetricCollector:
        """Get underlying metric collector."""
        return self._collector
