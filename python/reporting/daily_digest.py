"""
Daily Digest Generator
Implements an async daily digest generator summarizing PnL, Sharpe, and regime performance.
Compresses and rotates old log files to prevent the local disk from filling up over 24/7 runs.
"""

import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import gzip
import shutil
import os
import json

# Conditional aiofiles import
try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False
    aiofiles = None  # type: ignore


logger = logging.getLogger(__name__)


@dataclass
class DailyMetrics:
    """Daily performance metrics."""
    date: str
    total_pnl: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    avg_trade_duration_ms: int
    total_volume: float
    best_trade: float
    worst_trade: float


@dataclass
class RegimePerformance:
    """Performance broken down by market regime."""
    regime_name: str
    duration_minutes: int
    pnl_contribution: float
    trade_count: int
    avg_volatility: float


@dataclass
class DailyDigest:
    """Complete daily digest structure."""
    generated_at: str
    trading_date: str
    metrics: DailyMetrics
    regime_breakdown: List[RegimePerformance]
    hourly_pnl: Dict[str, float]
    top_instruments: List[Dict[str, Any]]
    risk_summary: Dict[str, Any]
    anomalies: List[str]


class DailyDigestGenerator:
    """
    Async daily digest generator with log rotation and compression.
    Summarizes PnL, Sharpe, regime performance for each trading day.
    """

    def __init__(
        self,
        output_directory: str = "./digests",
        log_directory: str = "./logs",
        retention_days: int = 30,
        compress_after_days: int = 7,
        max_log_size_mb: int = 100,
    ):
        """
        Initialize the digest generator.

        Args:
            output_directory: Directory for digest files
            log_directory: Directory for log files
            retention_days: Days to keep uncompressed logs
            compress_after_days: Days after which to compress logs
            max_log_size_mb: Maximum log file size before rotation
        """
        self.output_directory = output_directory
        self.log_directory = log_directory
        self.retention_days = retention_days
        self.compress_after_days = compress_after_days
        self.max_log_size_bytes = max_log_size_mb * 1024 * 1024

        self._running = False
        self._scheduled_digest_time: Optional[str] = None
        self._digest_history: List[str] = []

        # Ensure directories exist
        os.makedirs(output_directory, exist_ok=True)
        os.makedirs(log_directory, exist_ok=True)

    async def start(self, scheduled_time: str = "23:59") -> None:
        """
        Start the digest generator with scheduled daily generation.

        Args:
            scheduled_time: Time to generate digest (HH:MM format)
        """
        self._running = True
        self._scheduled_digest_time = scheduled_time
        logger.info(f"DailyDigestGenerator started, scheduled at {scheduled_time}")

        # Start the scheduling loop
        asyncio.create_task(self._schedule_loop())

    async def stop(self) -> None:
        """Stop the digest generator."""
        self._running = False
        logger.info("DailyDigestGenerator stopped")

    async def _schedule_loop(self) -> None:
        """Loop that checks for scheduled digest time."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute

                now = datetime.now()
                current_time = now.strftime("%H:%M")

                if current_time == self._scheduled_digest_time:
                    # Generate digest for today
                    await self.generate_daily_digest(now.date().isoformat())

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in schedule loop: {e}")

    async def generate_daily_digest(
        self,
        trading_date: str,
        metrics: Optional[DailyMetrics] = None,
    ) -> str:
        """
        Generate a daily digest for the specified trading date.

        Args:
            trading_date: Date string (YYYY-MM-DD)
            metrics: Pre-computed metrics (or will be calculated)

        Returns:
            Path to generated digest file
        """
        # Gather or compute metrics
        if metrics is None:
            metrics = await self._gather_daily_metrics(trading_date)

        # Generate regime breakdown
        regime_breakdown = await self._analyze_regimes(trading_date)

        # Generate hourly PnL
        hourly_pnl = await self._calculate_hourly_pnl(trading_date)

        # Get top instruments
        top_instruments = await self._get_top_instruments(trading_date)

        # Risk summary
        risk_summary = await self._generate_risk_summary(trading_date)

        # Detect anomalies
        anomalies = self._detect_anomalies(metrics, regime_breakdown)

        # Create digest
        digest = DailyDigest(
            generated_at=datetime.now().isoformat(),
            trading_date=trading_date,
            metrics=metrics,
            regime_breakdown=regime_breakdown,
            hourly_pnl=hourly_pnl,
            top_instruments=top_instruments,
            risk_summary=risk_summary,
            anomalies=anomalies,
        )

        # Save digest
        filepath = await self._save_digest(digest)
        self._digest_history.append(filepath)

        logger.info(f"Generated daily digest for {trading_date}: {filepath}")
        return filepath

    async def _gather_daily_metrics(self, trading_date: str) -> DailyMetrics:
        """Gather daily metrics from data sources."""
        # Placeholder - in production would query actual trade database
        # For now, generate synthetic metrics
        return DailyMetrics(
            date=trading_date,
            total_pnl=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            avg_trade_duration_ms=0,
            total_volume=0.0,
            best_trade=0.0,
            worst_trade=0.0,
        )

    async def _analyze_regimes(self, trading_date: str) -> List[RegimePerformance]:
        """Analyze performance by market regime."""
        # Placeholder - would integrate with regime detection system
        return []

    async def _calculate_hourly_pnl(self, trading_date: str) -> Dict[str, float]:
        """Calculate PnL broken down by hour."""
        # Placeholder
        return {}

    async def _get_top_instruments(self, trading_date: str) -> List[Dict[str, Any]]:
        """Get top performing instruments for the day."""
        # Placeholder
        return []

    async def _generate_risk_summary(self, trading_date: str) -> Dict[str, Any]:
        """Generate risk summary for the day."""
        return {
            "max_position_size": 0.0,
            "var_95": 0.0,
            "stress_test_result": "PASS",
            "risk_events": 0,
        }

    def _detect_anomalies(
        self,
        metrics: DailyMetrics,
        regimes: List[RegimePerformance],
    ) -> List[str]:
        """Detect trading anomalies."""
        anomalies = []

        if metrics.win_rate < 0.3 and metrics.total_trades > 10:
            anomalies.append(f"Low win rate: {metrics.win_rate:.1%}")

        if metrics.max_drawdown > 0.05:
            anomalies.append(f"High drawdown: {metrics.max_drawdown:.1%}")

        if abs(metrics.total_pnl) > 100000:  # $100k threshold
            anomalies.append(f"Large PnL day: ${metrics.total_pnl:,.2f}")

        return anomalies

    async def _save_digest(self, digest: DailyDigest) -> str:
        """Save digest to file asynchronously."""
        filename = f"daily_digest_{digest.trading_date}.json"
        filepath = f"{self.output_directory}/{filename}"

        # Convert to dict
        digest_dict = {
            "generated_at": digest.generated_at,
            "trading_date": digest.trading_date,
            "metrics": digest.metrics.__dict__,
            "regime_breakdown": [r.__dict__ for r in digest.regime_breakdown],
            "hourly_pnl": digest.hourly_pnl,
            "top_instruments": digest.top_instruments,
            "risk_summary": digest.risk_summary,
            "anomalies": digest.anomalies,
        }

        content = json.dumps(digest_dict, indent=2)

        if AIOFILES_AVAILABLE:
            async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
                await f.write(content)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            await asyncio.sleep(0)

        return filepath

    async def rotate_logs(self) -> List[str]:
        """
        Rotate and compress old log files.
        Returns list of processed files.
        """
        processed_files = []
        cutoff_compress = datetime.now() - timedelta(days=self.compress_after_days)
        cutoff_delete = datetime.now() - timedelta(days=self.retention_days)

        try:
            for filename in os.listdir(self.log_directory):
                filepath = os.path.join(self.log_directory, filename)

                if not os.path.isfile(filepath):
                    continue

                # Get file modification time
                mtime = datetime.fromtimestamp(os.path.getmtime(filepath))

                # Delete old files
                if mtime < cutoff_delete:
                    os.remove(filepath)
                    processed_files.append(f"Deleted: {filename}")
                    continue

                # Compress older files
                if mtime < cutoff_compress and not filename.endswith('.gz'):
                    compressed_path = f"{filepath}.gz"
                    with open(filepath, 'rb') as f_in:
                        with gzip.open(compressed_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    os.remove(filepath)
                    processed_files.append(f"Compressed: {filename}")

        except Exception as e:
            logger.error(f"Log rotation failed: {e}")

        return processed_files

    async def check_log_sizes(self) -> Dict[str, int]:
        """Check sizes of log files and return those exceeding limit."""
        oversized = {}

        try:
            for filename in os.listdir(self.log_directory):
                filepath = os.path.join(self.log_directory, filename)

                if os.path.isfile(filepath):
                    size = os.path.getsize(filepath)
                    if size > self.max_log_size_bytes:
                        oversized[filename] = size

        except Exception as e:
            logger.error(f"Log size check failed: {e}")

        return oversized

    async def cleanup_and_rotate(self) -> Dict[str, Any]:
        """
        Perform full cleanup: rotate logs, compress old files, delete expired.
        Returns summary of actions taken.
        """
        result = {
            "rotated_files": [],
            "oversized_files": {},
            "disk_usage_bytes": 0,
        }

        # Rotate and compress
        result["rotated_files"] = await self.rotate_logs()

        # Check for oversized files
        result["oversized_files"] = await self.check_log_sizes()

        # Calculate total disk usage
        total_size = 0
        try:
            for filename in os.listdir(self.log_directory):
                filepath = os.path.join(self.log_directory, filename)
                if os.path.isfile(filepath):
                    total_size += os.path.getsize(filepath)
        except Exception:
            pass

        result["disk_usage_bytes"] = total_size

        logger.info(
            f"Cleanup complete: {len(result['rotated_files'])} files processed, "
            f"{total_size / 1024 / 1024:.1f}MB total log size"
        )

        return result

    def get_digest_history(self) -> List[str]:
        """Get list of generated digest files."""
        return self._digest_history.copy()

    async def generate_digest_report(
        self,
        start_date: str,
        end_date: str,
    ) -> Dict[str, Any]:
        """
        Generate aggregated report across multiple days.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Aggregated metrics dictionary
        """
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)

        total_pnl = 0.0
        total_trades = 0
        total_winning = 0
        total_losing = 0
        daily_sharpes = []
        daily_drawdowns = []

        current = start
        while current <= end:
            date_str = current.date().isoformat()
            metrics = await self._gather_daily_metrics(date_str)

            total_pnl += metrics.total_pnl
            total_trades += metrics.total_trades
            total_winning += metrics.winning_trades
            total_losing += metrics.losing_trades
            if metrics.sharpe_ratio != 0:
                daily_sharpes.append(metrics.sharpe_ratio)
            if metrics.max_drawdown != 0:
                daily_drawdowns.append(metrics.max_drawdown)

            current += timedelta(days=1)

        return {
            "period": f"{start_date} to {end_date}",
            "total_pnl": total_pnl,
            "total_trades": total_trades,
            "win_rate": (total_winning / total_trades) if total_trades > 0 else 0,
            "avg_daily_sharpe": sum(daily_sharpes) / len(daily_sharpes) if daily_sharpes else 0,
            "max_drawdown": max(daily_drawdowns) if daily_drawdowns else 0,
            "days_traded": (end - start).days + 1,
        }
