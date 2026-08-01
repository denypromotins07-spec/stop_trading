#!/usr/bin/env python3
"""
PNL Tracker - Stage 50
Tracks real-time progress against financial goal (3,000 to 50,000 INR in 4 hours).
Uses real-time USDT/INR feeds for accurate conversion.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from pathlib import Path
import threading
import queue
import zmq
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)
logger = logging.getLogger('PNLTracker')

# Constants
INITIAL_CAPITAL_INR = 3000
TARGET_CAPITAL_INR = 50000
TRADING_WINDOW_HOURS = 4
USDT_INR_FEED_URL = "tcp://localhost:5562"


class USDTINRFeed:
    """Fetches real-time USDT/INR exchange rate."""
    
    def __init__(self):
        self.zmq_context = zmq.Context()
        self.feed_socket = self.zmq_context.socket(zmq.SUB)
        self.feed_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.latest_rate: float = 83.50  # Default fallback rate
        self.rate_timestamp: Optional[datetime] = None
        self.running = False
    
    def connect(self):
        """Connect to USDT/INR feed."""
        try:
            self.feed_socket.connect(USDT_INR_FEED_URL)
            logger.info(f"Connected to USDT/INR feed: {USDT_INR_FEED_URL}")
        except Exception as e:
            logger.warning(f"Could not connect to USDT/INR feed, using default rate: {self.latest_rate}")
    
    def start(self):
        """Start background rate update thread."""
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
    
    def _update_loop(self):
        """Background loop updating exchange rate."""
        poller = zmq.Poller()
        poller.register(self.feed_socket, zmq.POLLIN)
        
        while self.running:
            try:
                socks = dict(poller.poll(timeout=1000))
                
                if self.feed_socket in socks:
                    message = self.feed_socket.recv_json(flags=zmq.NOBLOCK)
                    rate = message.get('rate', self.latest_rate)
                    
                    if rate > 0:
                        self.latest_rate = rate
                        self.rate_timestamp = datetime.now()
                        
            except Exception as e:
                pass  # Silent fail, keep last known rate
        
        # Fallback: periodically fetch from external API if no feed
        if not self.rate_timestamp or (datetime.now() - self.rate_timestamp).total_seconds() > 60:
            self._fetch_fallback_rate()
    
    def _fetch_fallback_rate(self):
        """Fetch fallback rate from external API."""
        try:
            import urllib.request
            # Using a public API for USDT/INR (in production, use professional feed)
            url = "https://api.exchangerate-api.com/v4/latest/USD"
            
            with urllib.request.urlopen(url, timeout=5) as response:
                data = json.loads(response.read().decode())
                usd_inr = data.get('rates', {}).get('INR', 83.5)
                
                # USDT ≈ USD, so USDT/INR ≈ USD/INR
                self.latest_rate = usd_inr
                self.rate_timestamp = datetime.now()
                
                logger.debug(f"Fetched fallback USDT/INR rate: {self.latest_rate}")
        
        except Exception as e:
            logger.debug(f"Fallback rate fetch failed: {e}")
    
    def get_rate(self) -> float:
        """Get current USDT/INR rate."""
        return self.latest_rate
    
    def stop(self):
        """Stop feed updates."""
        self.running = False
        self.feed_socket.close()
        self.zmq_context.term()


class PNLCalculator:
    """Calculates P&L metrics and projections."""
    
    def __init__(self, initial_capital_inr: float = INITIAL_CAPITAL_INR):
        self.initial_capital_inr = initial_capital_inr
        self.current_capital_inr = initial_capital_inr
        self.realized_pnl_inr = 0.0
        self.unrealized_pnl_inr = 0.0
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.trades: list = []
        self.start_time: Optional[datetime] = None
        self._lock = threading.Lock()
    
    def start_session(self):
        """Start trading session."""
        self.start_time = datetime.now()
        logger.info(f"PNL session started at {self.start_time}")
    
    def record_trade(self, symbol: str, side: str, quantity: float, 
                     entry_price_usdt: float, exit_price_usdt: Optional[float] = None,
                     pnl_usdt: float = 0.0):
        """Record a trade."""
        with self._lock:
            self.trade_count += 1
            
            trade = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'entry_price_usdt': entry_price_usdt,
                'exit_price_usdt': exit_price_usdt,
                'pnl_usdt': pnl_usdt
            }
            self.trades.append(trade)
            
            if pnl_usdt != 0:
                self.realized_pnl_inr += pnl_usdt  # Will convert with rate
                if pnl_usdt > 0:
                    self.win_count += 1
                else:
                    self.loss_count += 1
    
    def update_unrealized_pnl(self, unrealized_usdt: float):
        """Update unrealized P&L."""
        with self._lock:
            self.unrealized_pnl_inr = unrealized_usdt
    
    def get_metrics(self, usdt_inr_rate: float) -> Dict:
        """Get current P&L metrics."""
        with self._lock:
            realized_inr = self.realized_pnl_inr * usdt_inr_rate
            unrealized_inr = self.unrealized_pnl_inr * usdt_inr_rate
            total_inr = self.initial_capital_inr + realized_inr + unrealized_inr
            
            elapsed = datetime.now() - self.start_time if self.start_time else timedelta(0)
            remaining = timedelta(hours=TRADING_WINDOW_HOURS) - elapsed
            
            # Calculate required rate to hit target
            target_remaining = TARGET_CAPITAL_INR - total_inr
            hours_remaining = remaining.total_seconds() / 3600
            required_hourly_rate = (target_remaining / total_inr / max(hours_remaining, 0.1)) * 100
            
            # Win rate
            win_rate = self.win_count / max(self.trade_count, 1) * 100
            
            return {
                'initial_capital_inr': self.initial_capital_inr,
                'current_capital_inr': total_inr,
                'realized_pnl_inr': realized_inr,
                'unrealized_pnl_inr': unrealized_inr,
                'total_pnl_inr': realized_inr + unrealized_inr,
                'total_pnl_percent': ((total_inr - self.initial_capital_inr) / self.initial_capital_inr) * 100,
                'target_capital_inr': TARGET_CAPITAL_INR,
                'remaining_to_target': max(0, TARGET_CAPITAL_INR - total_inr),
                'progress_percent': (total_inr / TARGET_CAPITAL_INR) * 100,
                'elapsed_minutes': elapsed.total_seconds() / 60,
                'remaining_minutes': max(0, remaining.total_seconds() / 60),
                'required_hourly_return_pct': required_hourly_rate,
                'trade_count': self.trade_count,
                'win_count': self.win_count,
                'loss_count': self.loss_count,
                'win_rate': win_rate,
                'usdt_inr_rate': usdt_inr_rate
            }
    
    def get_eta_to_target(self, usdt_inr_rate: float) -> Optional[timedelta]:
        """Estimate time to reach target based on current rate."""
        metrics = self.get_metrics(usdt_inr_rate)
        
        if metrics['elapsed_minutes'] < 5:
            return None  # Not enough data
        
        current_rate_per_hour = metrics['total_pnl_inr'] / max(metrics['elapsed_minutes'] / 60, 0.1)
        
        if current_rate_per_hour <= 0:
            return None  # Can't project negative/zero growth
        
        remaining = metrics['remaining_to_target']
        hours_needed = remaining / max(current_rate_per_hour, 1)
        
        return timedelta(hours=hours_needed)


class PNLTracker:
    """Main P&L tracking manager."""
    
    def __init__(self):
        self.feed = USDTINRFeed()
        self.calculator = PNLCalculator()
        self.running = False
        self.subscribers: list = []
    
    def start(self):
        """Start P&L tracking."""
        self.feed.connect()
        self.feed.start()
        self.calculator.start_session()
        self.running = True
        
        logger.info(f"PNL Tracker started")
        logger.info(f"Initial Capital: ₹{self.calculator.initial_capital_inr:,.2f}")
        logger.info(f"Target Capital: ₹{TARGET_CAPITAL_INR:,.2f}")
    
    def subscribe(self, callback):
        """Subscribe to P&L updates."""
        self.subscribers.append(callback)
    
    def notify_subscribers(self, metrics: Dict):
        """Notify all subscribers of P&L update."""
        for callback in self.subscribers:
            try:
                callback(metrics)
            except Exception as e:
                logger.error(f"Subscriber notification error: {e}")
    
    def get_current_metrics(self) -> Dict:
        """Get current P&L metrics."""
        rate = self.feed.get_rate()
        return self.calculator.get_metrics(rate)
    
    def get_progress_bar_data(self) -> Dict:
        """Get data for progress bar rendering."""
        metrics = self.get_current_metrics()
        
        progress = metrics['progress_percent']
        filled = int(30 * progress / 100)
        bar = "█" * filled + "░" * (30 - filled)
        
        eta = self.calculator.get_eta_to_target(self.feed.get_rate())
        eta_str = str(eta)[:7] if eta else "N/A"
        
        return {
            'bar': bar,
            'progress': progress,
            'current': metrics['current_capital_inr'],
            'target': TARGET_CAPITAL_INR,
            'remaining': metrics['remaining_to_target'],
            'eta': eta_str,
            'required_hourly_return': metrics['required_hourly_return_pct']
        }
    
    def record_trade(self, **kwargs):
        """Record a trade."""
        self.calculator.record_trade(**kwargs)
        self.notify_subscribers(self.get_current_metrics())
    
    def stop(self):
        """Stop P&L tracking."""
        self.running = False
        self.feed.stop()
        
        # Final metrics
        metrics = self.get_current_metrics()
        logger.info("=" * 60)
        logger.info("FINAL P&L SUMMARY")
        logger.info(f"Final Capital: ₹{metrics['current_capital_inr']:,.2f}")
        logger.info(f"Total P&L: ₹{metrics['total_pnl_inr']:,.2f} ({metrics['total_pnl_percent']:.2f}%)")
        logger.info(f"Trades: {metrics['trade_count']} (Win Rate: {metrics['win_rate']:.1f}%)")
        logger.info("=" * 60)


def main():
    """Entry point for P&L tracker testing."""
    tracker = PNLTracker()
    
    try:
        tracker.start()
        
        # Simulate some trades for testing
        import time
        
        for i in range(5):
            time.sleep(1)
            metrics = tracker.get_current_metrics()
            print(f"\rProgress: ₹{metrics['current_capital_inr']:,.2f} / ₹{TARGET_CAPITAL_INR:,.2f} "
                  f"({metrics['progress_percent']:.1f}%)", end="", flush=True)
        
        print()
        
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        tracker.stop()


if __name__ == '__main__':
    main()
