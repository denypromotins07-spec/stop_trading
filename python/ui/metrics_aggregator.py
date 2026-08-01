"""
Metrics Aggregator - Aggregates Ray worker health, Nautilus portfolio state, and ML inference latencies.
Produces unified, bounded JSON payload for UI consumption.
Memory-efficient with strict size limits.
"""
import numpy as np
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import time
import json
from collections import deque
from dataclasses import dataclass, asdict
import threading

logger = logging.getLogger(__name__)


@dataclass
class WorkerHealth:
    """Worker health metrics."""
    worker_id: str
    status: str  # 'healthy', 'degraded', 'unhealthy'
    cpu_percent: float
    memory_mb: float
    tasks_completed: int
    last_heartbeat: float


@dataclass
class PortfolioState:
    """Nautilus portfolio state snapshot."""
    total_value: float
    cash_balance: float
    positions_count: int
    unrealized_pnl: float
    realized_pnl: float
    exposure: float
    risk_metrics: Dict[str, float]


@dataclass
class InferenceMetrics:
    """ML inference latency metrics."""
    model_name: str
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    requests_per_second: float
    error_rate: float


class BoundedMetricsStore:
    """
    Thread-safe bounded storage for metrics history.
    Automatically prunes old entries to maintain memory limits.
    """
    
    def __init__(self, max_entries: int = 1000):
        self.max_entries = max_entries
        self._store: deque = deque(maxlen=max_entries)
        self._lock = threading.Lock()
    
    def append(self, entry: Dict[str, Any]) -> None:
        """Add entry to store."""
        with self._lock:
            self._store.append(entry)
    
    def get_recent(self, n: int = 100) -> List[Dict]:
        """Get n most recent entries."""
        with self._lock:
            return list(self._store)[-n:]
    
    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._store.clear()
    
    def __len__(self) -> int:
        return len(self._store)


class MetricsAggregator:
    """
    Central aggregator for all system metrics.
    Produces bounded JSON payloads for UI consumption.
    """
    
    MAX_PAYLOAD_SIZE = 1024 * 1024  # 1MB max payload
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Bounded stores for history
        self._worker_history = BoundedMetricsStore(
            max_entries=self.config.get('max_worker_history', 100)
        )
        self._portfolio_history = BoundedMetricsStore(
            max_entries=self.config.get('max_portfolio_history', 1000)
        )
        self._inference_history = BoundedMetricsStore(
            max_entries=self.config.get('max_inference_history', 500)
        )
        
        # Current state
        self._workers: Dict[str, WorkerHealth] = {}
        self._portfolio: Optional[PortfolioState] = None
        self._inference_stats: Dict[str, InferenceMetrics] = {}
        
        # Latency tracking
        self._latency_samples: deque = deque(maxlen=10000)
        
        # Aggregation timestamp
        self._last_aggregation = 0.0
        self._aggregation_interval = self.config.get('aggregation_interval', 1.0)
        
        # Payload cache
        self._cached_payload: Optional[str] = None
        self._cache_valid_until = 0.0
        
        logger.info("MetricsAggregator initialized")
    
    def update_worker_health(self, worker_id: str, status: str,
                             cpu_percent: float, memory_mb: float,
                             tasks_completed: int) -> None:
        """Update worker health metrics."""
        health = WorkerHealth(
            worker_id=worker_id,
            status=status,
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            tasks_completed=tasks_completed,
            last_heartbeat=time.time()
        )
        
        self._workers[worker_id] = health
        self._worker_history.append(asdict(health))
    
    def update_portfolio_state(self, total_value: float, cash_balance: float,
                                positions_count: int, unrealized_pnl: float,
                                realized_pnl: float, exposure: float,
                                risk_metrics: Optional[Dict] = None) -> None:
        """Update Nautilus portfolio state."""
        self._portfolio = PortfolioState(
            total_value=total_value,
            cash_balance=cash_balance,
            positions_count=positions_count,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            exposure=exposure,
            risk_metrics=risk_metrics or {}
        )
        
        self._portfolio_history.append(asdict(self._portfolio))
    
    def record_inference_latency(self, model_name: str, latency_ms: float,
                                  success: bool = True) -> None:
        """Record single inference latency sample."""
        self._latency_samples.append({
            'model': model_name,
            'latency_ms': latency_ms,
            'success': success,
            'timestamp': time.time()
        })
        
        # Update aggregated stats
        self._update_inference_stats(model_name)
    
    def _update_inference_stats(self, model_name: str) -> None:
        """Update inference statistics for a model."""
        # Get samples for this model
        model_samples = [
            s for s in self._latency_samples 
            if s['model'] == model_name
        ]
        
        if not model_samples:
            return
        
        latencies = [s['latency_ms'] for s in model_samples]
        successes = sum(1 for s in model_samples if s['success'])
        
        # Calculate percentiles
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)
        
        p50 = latencies_sorted[int(n * 0.50)] if n > 0 else 0
        p95 = latencies_sorted[int(n * 0.95)] if n > 0 else 0
        p99 = latencies_sorted[int(n * 0.99)] if n > 0 else 0
        
        # Calculate RPS (requests in last second)
        one_second_ago = time.time() - 1.0
        recent_requests = sum(1 for s in model_samples if s['timestamp'] > one_second_ago)
        
        # Error rate
        error_rate = 1.0 - (successes / max(1, len(model_samples)))
        
        self._inference_stats[model_name] = InferenceMetrics(
            model_name=model_name,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            requests_per_second=float(recent_requests),
            error_rate=error_rate
        )
        
        self._inference_history.append(asdict(self._inference_stats[model_name]))
    
    def aggregate(self, force: bool = False) -> Dict[str, Any]:
        """
        Aggregate all metrics into unified payload.
        
        Args:
            force: Force aggregation even if cache is valid
            
        Returns:
            Unified metrics dictionary
        """
        current_time = time.time()
        
        # Return cached payload if still valid
        if not force and current_time < self._cache_valid_until and self._cached_payload:
            return json.loads(self._cached_payload)
        
        # System overview
        system = {
            'timestamp': current_time,
            'uptime_seconds': current_time - (self._last_aggregation or current_time),
            'aggregation_version': 1
        }
        
        # Worker health summary
        workers_summary = {
            'total_workers': len(self._workers),
            'healthy_workers': sum(1 for w in self._workers.values() if w.status == 'healthy'),
            'degraded_workers': sum(1 for w in self._workers.values() if w.status == 'degraded'),
            'unhealthy_workers': sum(1 for w in self._workers.values() if w.status == 'unhealthy'),
            'avg_cpu_percent': np.mean([w.cpu_percent for w in self._workers.values()]) if self._workers else 0,
            'avg_memory_mb': np.mean([w.memory_mb for w in self._workers.values()]) if self._workers else 0,
            'workers': [asdict(w) for w in self._workers.values()]
        }
        
        # Portfolio summary
        if self._portfolio:
            portfolio_summary = asdict(self._portfolio)
        else:
            portfolio_summary = {
                'total_value': 0.0,
                'cash_balance': 0.0,
                'positions_count': 0,
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0,
                'exposure': 0.0,
                'risk_metrics': {}
            }
        
        # Inference summary
        inference_summary = {
            'models': {k: asdict(v) for k, v in self._inference_stats.items()},
            'total_models': len(self._inference_stats),
            'overall_p99_latency_ms': max(
                (v.p99_latency_ms for v in self._inference_stats.values()),
                default=0.0
            )
        }
        
        # Build aggregated payload
        aggregated = {
            'system': system,
            'workers': workers_summary,
            'portfolio': portfolio_summary,
            'inference': inference_summary,
            'history_counts': {
                'worker_samples': len(self._worker_history),
                'portfolio_samples': len(self._portfolio_history),
                'inference_samples': len(self._inference_history)
            }
        }
        
        # Validate payload size
        payload_json = json.dumps(aggregated)
        if len(payload_json.encode('utf-8')) > self.MAX_PAYLOAD_SIZE:
            logger.warning(f"Metrics payload exceeds limit: {len(payload_json)} bytes")
            # Trim history counts to reduce size
            aggregated['history_counts'] = {'trimmed': True}
            payload_json = json.dumps(aggregated)
        
        # Update cache
        self._cached_payload = payload_json
        self._cache_valid_until = current_time + self._aggregation_interval
        self._last_aggregation = current_time
        
        return aggregated
    
    def get_payload_json(self, force: bool = False) -> str:
        """Get aggregated metrics as JSON string."""
        if force or not self._cached_payload or time.time() >= self._cache_valid_until:
            self.aggregate(force=True)
        
        return self._cached_payload or '{}'
    
    def get_summary(self) -> Dict[str, Any]:
        """Get quick summary without full aggregation."""
        return {
            'workers': len(self._workers),
            'portfolio_value': self._portfolio.total_value if self._portfolio else 0,
            'models_tracked': len(self._inference_stats),
            'last_update': self._last_aggregation
        }
    
    def reset(self) -> None:
        """Reset all metrics."""
        self._workers.clear()
        self._portfolio = None
        self._inference_stats.clear()
        self._latency_samples.clear()
        self._worker_history.clear()
        self._portfolio_history.clear()
        self._inference_history.clear()
        self._cached_payload = None
        logger.info("MetricsAggregator reset")


# Singleton instance
_metrics_aggregator: Optional[MetricsAggregator] = None


def get_metrics_aggregator(config: Optional[Dict[str, Any]] = None) -> MetricsAggregator:
    """Get or create singleton MetricsAggregator instance."""
    global _metrics_aggregator
    if _metrics_aggregator is None:
        _metrics_aggregator = MetricsAggregator(config)
    return _metrics_aggregator


def reset_metrics_aggregator() -> None:
    """Reset singleton instance."""
    global _metrics_aggregator
    if _metrics_aggregator is not None:
        _metrics_aggregator.reset()
    _metrics_aggregator = None


__all__ = [
    'MetricsAggregator',
    'BoundedMetricsStore',
    'WorkerHealth',
    'PortfolioState',
    'InferenceMetrics',
    'get_metrics_aggregator',
    'reset_metrics_aggregator'
]
