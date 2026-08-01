"""
Reporting Module Root
Manages asynchronous file writes to prevent blocking the main ML inference loop.
Coordinates SOUL synthesis and daily digest generation.
"""

import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging
import os

# Import reporting components
try:
    from .soul_synthesizer import SOULSynthesizer, SOULReport, TradeOutcome, MLMetrics, StrategyPerformance
    from .daily_digest import DailyDigestGenerator, DailyMetrics
except ImportError:
    from soul_synthesizer import SOULSynthesizer, SOULReport, TradeOutcome, MLMetrics, StrategyPerformance
    from daily_digest import DailyDigestGenerator, DailyMetrics


logger = logging.getLogger(__name__)


@dataclass
class ReportingConfig:
    """Configuration for reporting module."""
    output_directory: str = "./reports"
    digest_directory: str = "./digests"
    log_directory: str = "./logs"
    soul_template_path: Optional[str] = None
    digest_schedule: str = "23:59"
    retention_days: int = 30
    compress_after_days: int = 7


class ReportingModule:
    """
    Module root managing all reporting functionality.
    Coordinates async file I/O for SOUL.md synthesis and daily digests.
    Ensures disk writes never block the ML inference loop.
    """

    def __init__(
        self,
        config: Optional[ReportingConfig] = None,
    ):
        """
        Initialize the reporting module.

        Args:
            config: Reporting configuration
        """
        self.config = config or ReportingConfig()

        # Initialize components
        self.soul_synthesizer = SOULSynthesizer(
            output_directory=self.config.output_directory,
            template_string=None,  # Use default template
        )
        self.digest_generator = DailyDigestGenerator(
            output_directory=self.config.digest_directory,
            log_directory=self.config.log_directory,
            retention_days=self.config.retention_days,
            compress_after_days=self.config.compress_after_days,
        )

        # State
        self._running = False
        self._pending_reports: asyncio.Queue = asyncio.Queue()
        self._report_count = 0
        self._error_count = 0

    async def start(self) -> None:
        """Start the reporting module."""
        self._running = True

        # Start sub-components
        await self.soul_synthesizer.start()
        await self.digest_generator.start(scheduled_time=self.config.digest_schedule)

        # Start report processing loop
        asyncio.create_task(self._process_reports())

        logger.info("ReportingModule started")

    async def stop(self) -> None:
        """Stop the reporting module gracefully."""
        self._running = False

        # Wait for pending reports
        await self._pending_reports.join()

        # Stop sub-components
        await self.soul_synthesizer.stop()
        await self.digest_generator.stop()

        logger.info("ReportingModule stopped")

    async def _process_reports(self) -> None:
        """Process queued report generation requests."""
        while self._running:
            try:
                task = await asyncio.wait_for(
                    self._pending_reports.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            try:
                report_type, args = task
                if report_type == "soul":
                    await self._generate_soul_report(*args)
                elif report_type == "digest":
                    await self._generate_daily_digest(*args)
            except Exception as e:
                logger.error(f"Report generation failed: {e}")
                self._error_count += 1
            finally:
                self._pending_reports.task_done()

    async def queue_soul_report(
        self,
        report: SOULReport,
        filename: Optional[str] = None,
    ) -> None:
        """Queue a SOUL.md report for async generation."""
        await self._pending_reports.put(("soul", (report, filename)))
        logger.debug(f"Queued SOUL report: {filename or 'auto-generated'}")

    async def queue_daily_digest(
        self,
        trading_date: str,
    ) -> None:
        """Queue a daily digest for async generation."""
        await self._pending_reports.put(("digest", (trading_date,)))
        logger.debug(f"Queued daily digest for: {trading_date}")

    async def _generate_soul_report(
        self,
        report: SOULReport,
        filename: Optional[str],
    ) -> str:
        """Generate and save a SOUL.md report."""
        filepath = await self.soul_synthesizer.save_report(report, filename)
        self._report_count += 1
        logger.info(f"Generated SOUL report: {filepath}")
        return filepath

    async def _generate_daily_digest(
        self,
        trading_date: str,
    ) -> str:
        """Generate a daily digest."""
        filepath = await self.digest_generator.generate_daily_digest(trading_date)
        self._report_count += 1
        logger.info(f"Generated daily digest: {filepath}")
        return filepath

    async def generate_soul_report_immediate(
        self,
        report: SOULReport,
        filename: str,
    ) -> str:
        """Generate SOUL report immediately (bypasses queue)."""
        filepath = await self.soul_synthesizer.save_report_immediate(report, filename)
        self._report_count += 1
        return filepath

    async def perform_log_maintenance(self) -> Dict[str, Any]:
        """Perform log rotation and cleanup."""
        result = await self.digest_generator.cleanup_and_rotate()
        logger.info(
            f"Log maintenance complete: {len(result.get('rotated_files', []))} files processed"
        )
        return result

    def get_status(self) -> Dict[str, Any]:
        """Get reporting module status."""
        return {
            "running": self._running,
            "pending_reports": self._pending_reports.qsize(),
            "total_reports_generated": self._report_count,
            "error_count": self._error_count,
            "soul_history_length": len(self.soul_synthesizer._digest_history if hasattr(self.soul_synthesizer, '_digest_history') else []),
        }

    async def generate_end_of_day_report(
        self,
        trading_date: str,
        trades: List[TradeOutcome],
        ml_metrics: List[MLMetrics],
        strategies: List[StrategyPerformance],
        risk_events: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """
        Generate comprehensive end-of-day report including both SOUL.md and digest.

        Args:
            trading_date: Date string (YYYY-MM-DD)
            trades: List of trade outcomes
            ml_metrics: ML model metrics
            strategies: Strategy performance metrics
            risk_events: Risk events for the day

        Returns:
            Dictionary with paths to generated files
        """
        from datetime import datetime

        # Create SOUL report
        session_start = datetime.fromisoformat(f"{trading_date}T09:30:00")
        session_end = datetime.fromisoformat(f"{trading_date}T16:00:00")

        soul_report = SOULSynthesizer.create_soul_report(
            report_id=f"EOD_{trading_date}",
            session_start=session_start,
            session_end=session_end,
            trades=trades,
            ml_metrics=ml_metrics,
            strategies=strategies,
            risk_events=risk_events,
        )

        # Generate both reports
        soul_filename = f"soul_eod_{trading_date}.md"
        soul_path = await self.generate_soul_report_immediate(soul_report, soul_filename)
        digest_path = await self.digest_generator.generate_daily_digest(trading_date)

        return {
            "soul_report": soul_path,
            "daily_digest": digest_path,
        }


# Helper function to create SOULReport
def create_soul_report(
    report_id: str,
    session_start: Any,
    session_end: Any,
    trades: List[TradeOutcome],
    ml_metrics: List[MLMetrics],
    strategies: List[StrategyPerformance],
    risk_events: List[Dict[str, Any]],
) -> SOULReport:
    """Helper to create SOULReport using synthesizer's factory."""
    return SOULSynthesizer.create_soul_report(
        report_id=report_id,
        session_start=session_start,
        session_end=session_end,
        trades=trades,
        ml_metrics=ml_metrics,
        strategies=strategies,
        risk_events=risk_events,
    )


# Module-level singleton
_module_instance: Optional[ReportingModule] = None


def get_module() -> ReportingModule:
    """Get the module singleton instance."""
    global _module_instance
    if _module_instance is None:
        _module_instance = ReportingModule()
    return _module_instance


async def initialize_module(config: Optional[ReportingConfig] = None) -> ReportingModule:
    """Initialize the reporting module."""
    module = ReportingModule(config=config)
    await module.start()
    return module


async def shutdown_module() -> None:
    """Shutdown the reporting module."""
    module = get_module()
    await module.stop()
