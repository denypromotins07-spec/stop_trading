#!/usr/bin/env python3
"""
Global Dashboard - Stage 50
Aggregates Rust TUI data and Python ML metrics into unified rich Live dashboard.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, Optional, Any
from pathlib import Path
import threading
import queue
import zmq

# Try to import rich, fallback gracefully
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich.text import Text
    from rich.layout import Layout
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None
    Live = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)
logger = logging.getLogger('GlobalDashboard')

# Constants
ZMQ_TELEMETRY_URL = "tcp://localhost:5561"
DASHBOARD_REFRESH_RATE = 4  # FPS


class TelemetryCollector:
    """Collects telemetry data from Rust and Python components."""
    
    def __init__(self):
        self.zmq_context = zmq.Context()
        self.telemetry_socket = self.zmq_context.socket(zmq.SUB)
        self.telemetry_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.data_queue = queue.Queue(maxsize=1000)
        self.latest_data: Dict[str, Any] = {}
        self.running = False
        self._lock = threading.Lock()
    
    def connect(self):
        """Connect to telemetry socket."""
        try:
            self.telemetry_socket.connect(ZMQ_TELEMETRY_URL)
            logger.info(f"Connected to telemetry: {ZMQ_TELEMETRY_URL}")
        except Exception as e:
            logger.warning(f"Could not connect to telemetry: {e}")
    
    def start_collection(self):
        """Start background collection thread."""
        self.running = True
        self.collection_thread = threading.Thread(target=self._collect_loop, daemon=True)
        self.collection_thread.start()
    
    def _collect_loop(self):
        """Background loop collecting telemetry data."""
        poller = zmq.Poller()
        poller.register(self.telemetry_socket, zmq.POLLIN)
        
        while self.running:
            try:
                socks = dict(poller.poll(timeout=100))
                
                if self.telemetry_socket in socks:
                    message = self.telemetry_socket.recv_json(flags=zmq.NOBLOCK)
                    
                    # Update latest data with lock
                    with self._lock:
                        self.latest_data.update(message)
                        self.latest_data['last_update'] = datetime.now().isoformat()
                    
                    # Also queue for history
                    try:
                        self.data_queue.put_nowait(message)
                    except queue.Full:
                        # Drop oldest if queue full
                        try:
                            self.data_queue.get_nowait()
                            self.data_queue.put_nowait(message)
                        except:
                            pass
            
            except Exception as e:
                pass  # Silent fail for telemetry
    
    def get_latest(self) -> Dict:
        """Get latest telemetry data."""
        with self._lock:
            return self.latest_data.copy()
    
    def stop(self):
        """Stop collection."""
        self.running = False
        self.telemetry_socket.close()
        self.zmq_context.term()


class DashboardRenderer:
    """Renders the unified dashboard."""
    
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.layout: Optional[Layout] = None
        self._setup_layout()
    
    def _setup_layout(self):
        """Setup dashboard layout."""
        if not RICH_AVAILABLE or not self.console:
            return
        
        self.layout = Layout()
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        self.layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        self.layout["left"].split(
            Layout(name="system", ratio=2),
            Layout(name="strategies", ratio=2),
            Layout(name="ml_metrics", ratio=2)
        )
        self.layout["right"].split(
            Layout(name="pnl", ratio=2),
            Layout(name="positions", ratio=3),
            Layout(name="logs", ratio=2)
        )
    
    def render_header(self, data: Dict) -> Panel:
        """Render dashboard header."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        state = data.get('state', 'UNKNOWN')
        uptime = data.get('uptime', 'N/A')
        
        title_text = Text()
        title_text.append("🤖 CRYPTO MFT BOT ", style="bold cyan")
        title_text.append(f"| {state}", style="bold green" if state == 'RUNNING' else "bold yellow")
        title_text.append(f" | {now}", style="dim")
        
        return Panel(
            title_text,
            title="STAGE 50 - GLOBAL DASHBOARD",
            border_style="cyan"
        )
    
    def render_system_panel(self, data: Dict) -> Panel:
        """Render system metrics panel."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        rust_alive = data.get('rust_alive', False)
        python_alive = data.get('python_alive', False)
        rust_mem = data.get('rust_memory_mb', 0)
        python_mem = data.get('python_memory_mb', 0)
        
        table.add_row("Rust Core", f"{'✅' if rust_alive else '❌'} {rust_mem:.0f}MB")
        table.add_row("Python Daemon", f"{'✅' if python_alive else '❌'} {python_mem:.0f}MB")
        table.add_row("Trading Window", data.get('window_remaining', 'N/A'))
        table.add_row("Network Latency", f"{data.get('latency_ms', 0):.1f}ms")
        
        return Panel(table, title="[bold]System Status[/bold]", border_style="green")
    
    def render_pnl_panel(self, data: Dict) -> Panel:
        """Render P&L panel with progress to goal."""
        from .pnl_tracker import PNLTracker
        
        tracker = PNLTracker()
        current_inr = data.get('pnl_inr', 3000)
        target_inr = 50000
        progress_pct = (current_inr / target_inr) * 100
        
        pnl_text = Text()
        pnl_text.append(f"Current: ₹{current_inr:,.2f}\n", style="bold white")
        pnl_text.append(f"Target:  ₹{target_inr:,.2f}\n", style="dim")
        pnl_text.append(f"Progress: {progress_pct:.1f}%", style="bold green")
        
        # Add progress bar visualization
        bar_length = 30
        filled = int(bar_length * progress_pct / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        return Panel(
            f"[green]{bar}[/green]\n\n{pnl_text}",
            title="[bold]Financial Goal[/bold]",
            border_style="green" if current_inr >= 3000 else "red"
        )
    
    def render_footer(self, data: Dict) -> Panel:
        """Render dashboard footer."""
        last_update = data.get('last_update', 'N/A')
        fps = data.get('dashboard_fps', DASHBOARD_REFRESH_RATE)
        
        return Panel(
            f"Last update: {last_update} | Refresh: {fps} FPS | Press Ctrl+C for shutdown",
            title="Status",
            border_style="dim"
        )
    
    def generate_layout(self, data: Dict) -> Layout:
        """Generate complete dashboard layout."""
        if not RICH_AVAILABLE or not self.layout:
            return None
        
        self.layout["header"].update(self.render_header(data))
        self.layout["left"]["system"].update(self.render_system_panel(data))
        self.layout["right"]["pnl"].update(self.render_pnl_panel(data))
        self.layout["footer"].update(self.render_footer(data))
        
        return self.layout


class GlobalDashboard:
    """Main global dashboard manager."""
    
    def __init__(self):
        self.collector = TelemetryCollector()
        self.renderer = DashboardRenderer()
        self.running = False
        self.live_display: Optional[Live] = None
    
    def start(self):
        """Start the dashboard."""
        if not RICH_AVAILABLE:
            logger.warning("Rich not available, using text-only dashboard")
            self._run_text_dashboard()
            return
        
        self.collector.connect()
        self.collector.start_collection()
        
        self.running = True
        
        try:
            self.live_display = Live(
                self.renderer.generate_layout(self.collector.get_latest()),
                console=self.renderer.console,
                refresh_per_second=DASHBOARD_REFRESH_RATE,
                screen=True
            )
            
            with self.live_display:
                while self.running:
                    self.live_display.update(
                        self.renderer.generate_layout(self.collector.get_latest())
                    )
                    threading.Event().wait(1 / DASHBOARD_REFRESH_RATE)
        
        except KeyboardInterrupt:
            logger.info("Dashboard interrupted")
        finally:
            self.stop()
    
    def _run_text_dashboard(self):
        """Fallback text-only dashboard."""
        self.collector.connect()
        self.collector.start_collection()
        
        self.running = True
        
        try:
            while self.running:
                data = self.collector.get_latest()
                
                print("\n" + "=" * 60)
                print(f"CRYPTO MFT BOT - {datetime.now().strftime('%H:%M:%S')}")
                print("=" * 60)
                print(f"State: {data.get('state', 'UNKNOWN')}")
                print(f"Rust: {'✅' if data.get('rust_alive') else '❌'} | Python: {'✅' if data.get('python_alive') else '❌'}")
                print(f"P&L: ₹{data.get('pnl_inr', 3000):,.2f} / ₹50,000")
                print("=" * 60)
                
                threading.Event().wait(1)
        
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
    
    def stop(self):
        """Stop the dashboard."""
        self.running = False
        self.collector.stop()


def main():
    """Entry point for global dashboard."""
    dashboard = GlobalDashboard()
    
    try:
        dashboard.start()
    except KeyboardInterrupt:
        print("\nDashboard stopped")
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
