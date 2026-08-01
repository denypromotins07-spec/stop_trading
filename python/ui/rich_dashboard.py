"""
Rich Dashboard for Python Backend.
Backend dashboard streaming Ray cluster health, Nautilus state, and ML inference queues.
Pushes formatted strings to Rust TUI via dedicated ZMQ channel.

Non-blocking async updates ensure rendering never delays inference.
"""

import numpy as np
from typing import Dict, Any, Optional, List
import threading
import logging
import time
import json
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    import zmq
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClusterHealth:
    """Ray cluster health metrics."""
    active_workers: int
    total_workers: int
    cpu_utilization: float
    memory_utilization: float
    object_store_usage: float
    tasks_pending: int
    tasks_running: int
    status: str  # 'healthy', 'degraded', 'critical'


@dataclass
class NautilusState:
    """Nautilus trading engine state."""
    position: float
    unrealized_pnl: float
    realized_pnl: float
    orders_pending: int
    orders_filled_today: int
    risk_limit_remaining: float
    status: str  # 'running', 'paused', 'stopped'


@dataclass
class MLInferenceQueue:
    """ML inference queue metrics."""
    queue_depth: int
    avg_latency_us: float
    p99_latency_us: float
    requests_per_second: float
    model_version: str
    gpu_utilization: float


class RichDashboard:
    """
    Rich-based dashboard for monitoring system state.
    Pushes updates to Rust TUI via ZMQ.
    """
    
    def __init__(
        self,
        zmq_endpoint: str = "tcp://localhost:5555",
        update_interval_seconds: float = 1.0
    ):
        self.zmq_endpoint = zmq_endpoint
        self.update_interval = update_interval_seconds
        
        self._lock = threading.RLock()
        
        # State containers
        self._cluster_health: Optional[ClusterHealth] = None
        self._nautilus_state: Optional[NautilusState] = None
        self._ml_queue: Optional[MLInferenceQueue] = None
        self._custom_metrics: Dict[str, float] = {}
        
        # ZMQ socket for pushing to Rust TUI
        self._zmq_context: Optional[Any] = None
        self._zmq_socket: Optional[Any] = None
        
        # Rich console (for local display)
        self._console: Optional[Console] = None
        self._live: Optional[Live] = None
        
        # Control
        self._running = False
        self._update_thread: Optional[threading.Thread] = None
        
        if ZMQ_AVAILABLE:
            self._setup_zmq()
        
        if RICH_AVAILABLE:
            self._console = Console()
    
    def _setup_zmq(self) -> None:
        """Setup ZMQ publisher socket."""
        try:
            self._zmq_context = zmq.Context()
            self._zmq_socket = self._zmq_context.socket(zmq.PUB)
            self._zmq_socket.bind(self.zmq_endpoint)
            logger.info(f"ZMQ publisher bound to {self.zmq_endpoint}")
        except Exception as e:
            logger.warning(f"Failed to setup ZMQ: {e}")
            self._zmq_socket = None
    
    def update_cluster_health(self, health: ClusterHealth) -> None:
        """Update cluster health metrics."""
        with self._lock:
            self._cluster_health = health
    
    def update_nautilus_state(self, state: NautilusState) -> None:
        """Update Nautilus trading state."""
        with self._lock:
            self._nautilus_state = state
    
    def update_ml_queue(self, queue: MLInferenceQueue) -> None:
        """Update ML inference queue metrics."""
        with self._lock:
            self._ml_queue = queue
    
    def update_custom_metric(self, name: str, value: float) -> None:
        """Update a custom metric."""
        with self._lock:
            self._custom_metrics[name] = value
    
    def start(self) -> None:
        """Start dashboard update loop."""
        if self._running:
            return
        
        self._running = True
        
        def update_loop():
            while self._running:
                try:
                    self._render_and_push()
                except Exception as e:
                    logger.error(f"Dashboard update error: {e}")
                
                time.sleep(self.update_interval)
        
        self._update_thread = threading.Thread(target=update_loop, daemon=True)
        self._update_thread.start()
        logger.info("Rich Dashboard started")
    
    def stop(self) -> None:
        """Stop dashboard."""
        self._running = False
        if self._update_thread:
            self._update_thread.join(timeout=5)
        
        if self._live:
            self._live.stop()
        
        if self._zmq_socket:
            self._zmq_socket.close()
        if self._zmq_context:
            self._zmq_context.term()
        
        logger.info("Rich Dashboard stopped")
    
    def _render_and_push(self) -> None:
        """Render dashboard and push to ZMQ."""
        with self._lock:
            # Build dashboard data
            dashboard_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'cluster': asdict(self._cluster_health) if self._cluster_health else None,
                'nautilus': asdict(self._nautilus_state) if self._nautilus_state else None,
                'ml_queue': asdict(self._ml_queue) if self._ml_queue else None,
                'custom_metrics': self._custom_metrics.copy()
            }
            
            # Push to ZMQ
            if self._zmq_socket:
                try:
                    message = json.dumps(dashboard_data)
                    self._zmq_socket.send_string(message)
                except Exception as e:
                    logger.debug(f"ZMQ send error: {e}")
            
            # Local render if console available
            if self._console and RICH_AVAILABLE:
                self._render_local(dashboard_data)
    
    def _render_local(self, data: Dict[str, Any]) -> None:
        """Render dashboard locally using Rich."""
        if not RICH_AVAILABLE or not self._console:
            return
        
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        # Header
        header = Table(show_header=False, box=None)
        header.add_column(style="bold cyan")
        header.add_row(f"HFT System Dashboard - {data['timestamp'][:19]}")
        layout["header"].update(header)
        
        # Body - split into panels
        body = layout["body"]
        body.split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        # Cluster panel
        cluster_table = self._render_cluster_panel(data.get('cluster'))
        body["left"].split_column(
            Layout(name="cluster", ratio=2),
            Layout(name="nautilus", ratio=2),
            Layout(name="ml", ratio=1)
        )
        body["left"]["cluster"].update(cluster_table)
        
        # Nautilus panel
        nautilus_table = self._render_nautilus_panel(data.get('nautilus'))
        body["left"]["nautilus"].update(nautilus_table)
        
        # ML Queue panel
        ml_table = self._render_ml_panel(data.get('ml_queue'))
        body["left"]["ml"].update(ml_table)
        
        # Custom metrics panel
        custom_table = self._render_custom_metrics_panel(data.get('custom_metrics', {}))
        body["right"].update(custom_table)
        
        # Footer
        footer_text = Text("System Operational | Press Ctrl+C to exit", style="green")
        layout["footer"].update(Panel(footer_text))
        
        # Render
        try:
            self._console.print(layout)
        except Exception:
            pass  # Ignore render errors in non-interactive environments
    
    def _render_cluster_panel(self, cluster: Optional[Dict[str, Any]]) -> Table:
        """Render cluster health panel."""
        table = Table(title="Ray Cluster", box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        if cluster:
            status_color = "green" if cluster['status'] == 'healthy' else "red"
            table.add_row("Status", f"[{status_color}]{cluster['status']}[/{status_color}]")
            table.add_row("Workers", f"{cluster['active_workers']}/{cluster['total_workers']}")
            table.add_row("CPU", f"{cluster['cpu_utilization']:.1f}%")
            table.add_row("Memory", f"{cluster['memory_utilization']:.1f}%")
            table.add_row("Tasks Pending", str(cluster['tasks_pending']))
        else:
            table.add_row("Status", "[yellow]No data[/yellow]")
        
        return table
    
    def _render_nautilus_panel(self, nautilus: Optional[Dict[str, Any]]) -> Table:
        """Render Nautilus state panel."""
        table = Table(title="Nautilus Engine", box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        if nautilus:
            pnl_color = "green" if nautilus['unrealized_pnl'] >= 0 else "red"
            table.add_row("Position", f"{nautilus['position']:.2f}")
            table.add_row("Unrealized PnL", f"[{pnl_color}]${nautilus['unrealized_pnl']:.2f}[/{pnl_color}]")
            table.add_row("Realized PnL", f"${nautilus['realized_pnl']:.2f}")
            table.add_row("Orders Filled", str(nautilus['orders_filled_today']))
            table.add_row("Risk Remaining", f"${nautilus['risk_limit_remaining']:.2f}")
        else:
            table.add_row("Status", "[yellow]No data[/yellow]")
        
        return table
    
    def _render_ml_panel(self, ml: Optional[Dict[str, Any]]) -> Table:
        """Render ML inference panel."""
        table = Table(title="ML Inference", box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        if ml:
            table.add_row("Queue Depth", str(ml['queue_depth']))
            table.add_row("Avg Latency", f"{ml['avg_latency_us']:.0f} µs")
            table.add_row("P99 Latency", f"{ml['p99_latency_us']:.0f} µs")
            table.add_row("Throughput", f"{ml['requests_per_second']:.1f} req/s")
            table.add_row("Model", ml['model_version'])
        else:
            table.add_row("Status", "[yellow]No data[/yellow]")
        
        return table
    
    def _render_custom_metrics_panel(self, metrics: Dict[str, float]) -> Table:
        """Render custom metrics panel."""
        table = Table(title="Custom Metrics", box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        if metrics:
            for name, value in metrics.items():
                table.add_row(name, f"{value:.4f}")
        else:
            table.add_row("Status", "[yellow]No custom metrics[/yellow]")
        
        return table


# Global singleton instance
_dashboard_instance: Optional[RichDashboard] = None
_dashboard_lock = threading.Lock()


def get_dashboard(zmq_endpoint: str = "tcp://localhost:5555") -> RichDashboard:
    """Thread-safe singleton access to Rich dashboard."""
    global _dashboard_instance
    
    with _dashboard_lock:
        if _dashboard_instance is None:
            _dashboard_instance = RichDashboard(zmq_endpoint)
        
        return _dashboard_instance


if __name__ == "__main__":
    # Demo usage
    dashboard = get_dashboard()
    
    print("=== Rich Dashboard Demo ===\n")
    
    # Start dashboard
    dashboard.start()
    
    # Simulate updates
    for i in range(10):
        # Update cluster health
        dashboard.update_cluster_health(ClusterHealth(
            active_workers=4,
            total_workers=4,
            cpu_utilization=45 + np.random.randn() * 5,
            memory_utilization=60 + np.random.randn() * 3,
            object_store_usage=30,
            tasks_pending=np.random.randint(0, 10),
            tasks_running=np.random.randint(5, 20),
            status='healthy'
        ))
        
        # Update Nautilus state
        dashboard.update_nautilus_state(NautilusState(
            position=np.random.randn() * 10,
            unrealized_pnl=np.random.randn() * 1000,
            realized_pnl=5000 + np.random.randn() * 500,
            orders_pending=np.random.randint(0, 5),
            orders_filled_today=50 + i * 5,
            risk_limit_remaining=50000 - i * 1000,
            status='running'
        ))
        
        # Update ML queue
        dashboard.update_ml_queue(MLInferenceQueue(
            queue_depth=np.random.randint(0, 20),
            avg_latency_us=50 + np.random.exponential(10),
            p99_latency_us=200 + np.random.exponential(50),
            requests_per_second=100 + np.random.randn() * 20,
            model_version="v1.2.3",
            gpu_utilization=70 + np.random.randn() * 5
        ))
        
        # Update custom metrics
        dashboard.update_custom_metric("sharpe_ratio", 1.5 + np.random.randn() * 0.1)
        dashboard.update_custom_metric("drawdown", 0.05 + abs(np.random.randn()) * 0.01)
        
        time.sleep(1)
    
    dashboard.stop()
    print("\nDashboard demo complete")
