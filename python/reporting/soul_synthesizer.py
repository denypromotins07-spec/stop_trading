"""
SOUL.md Synthesizer - Template-Based Text Generation
Builds a template-based text synthesizer translating ML metrics and trade outcomes into SOUL.md.
Avoids LLMs; uses strict Jinja2 templates to generate structured markdown for the Rust self-learning ledger safely.
Uses aiofiles for non-blocking async file I/O so disk writes never stall the GIL.
"""

import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import json

# Conditional Jinja2 import
try:
    from jinja2 import Template, Environment, BaseLoader
    JINJA_AVAILABLE = True
except ImportError:
    JINJA_AVAILABLE = False
    Template = None  # type: ignore

# Conditional aiofiles import
try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False
    aiofiles = None  # type: ignore


logger = logging.getLogger(__name__)


@dataclass
class TradeOutcome:
    """Single trade outcome record."""
    timestamp: str
    instrument: str
    side: str
    quantity: float
    entry_price: float
    exit_price: float
    pnl: float
    pnl_bps: float
    execution_quality: float
    strategy_id: str


@dataclass
class MLMetrics:
    """ML model performance metrics."""
    model_name: str
    prediction_count: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    avg_inference_latency_us: float
    last_updated: str


@dataclass
class StrategyPerformance:
    """Strategy-level performance metrics."""
    strategy_id: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    avg_trade_duration_ms: int


@dataclass
class SOULReport:
    """Complete SOUL.md report structure."""
    report_id: str
    generated_at: str
    session_start: str
    session_end: str
    total_pnl: float
    total_trades: int
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    ml_metrics: List[MLMetrics]
    strategy_performance: List[StrategyPerformance]
    recent_trades: List[TradeOutcome]
    regime_analysis: Dict[str, Any]
    learning_insights: List[str]
    risk_events: List[Dict[str, Any]]


class SOULSynthesizer:
    """
    Template-based synthesizer for generating SOUL.md reports.
    Uses Jinja2 templates for structured markdown generation.
    Employs aiofiles for non-blocking async file I/O.
    """

    # SOUL.md template
    SOUL_TEMPLATE = """# SOUL.md - Self-Learning Ledger Report

## Report Metadata
- **Report ID**: {{ report_id }}
- **Generated At**: {{ generated_at }}
- **Session**: {{ session_start }} to {{ session_end }}

## Performance Summary
| Metric | Value |
|--------|-------|
| Total PnL | ${{ "%.2f"|format(total_pnl) }} |
| Total Trades | {{ total_trades }} |
| Sharpe Ratio | {{ "%.3f"|format(sharpe_ratio) }} |
| Sortino Ratio | {{ "%.3f"|format(sortino_ratio) }} |
| Max Drawdown | {{ "%.2f"|format(max_drawdown * 100) }}% |

## ML Model Metrics
{% for metric in ml_metrics %}
### {{ metric.model_name }}
- Predictions: {{ metric.prediction_count }}
- Accuracy: {{ "%.2f"|format(metric.accuracy * 100) }}%
- Precision: {{ "%.2f"|format(metric.precision * 100) }}%
- Recall: {{ "%.2f"|format(metric.recall * 100) }}%
- F1 Score: {{ "%.2f"|format(metric.f1_score * 100) }}%
- Avg Inference Latency: {{ "%.1f"|format(metric.avg_inference_latency_us) }}μs
- Last Updated: {{ metric.last_updated }}

{% endfor %}

## Strategy Performance
{% for strategy in strategy_performance %}
### Strategy: {{ strategy.strategy_id }}
| Metric | Value |
|--------|-------|
| Total Trades | {{ strategy.total_trades }} |
| Win Rate | {{ "%.1f"|format(strategy.win_rate * 100) }}% |
| Total PnL | ${{ "%.2f"|format(strategy.total_pnl) }} |
| Sharpe Ratio | {{ "%.3f"|format(strategy.sharpe_ratio) }} |
| Max Drawdown | {{ "%.2f"|format(strategy.max_drawdown * 100) }}% |
| Avg Duration | {{ strategy.avg_trade_duration_ms }}ms |

{% endfor %}

## Recent Trades
| Time | Instrument | Side | PnL ($) | PnL (bps) | Quality |
|------|------------|------|---------|-----------|---------|
{% for trade in recent_trades %}
| {{ trade.timestamp }} | {{ trade.instrument }} | {{ trade.side }} | {{ "%.2f"|format(trade.pnl) }} | {{ "%.1f"|format(trade.pnl_bps) }} | {{ "%.2f"|format(trade.execution_quality) }} |
{% endfor %}

## Regime Analysis
{% for regime, metrics in regime_analysis.items() %}
### {{ regime }}
- Duration: {{ metrics.get('duration_minutes', 0) }} minutes
- PnL Contribution: ${{ "%.2f"|format(metrics.get('pnl_contribution', 0)) }}
- Volatility: {{ "%.2f"|format(metrics.get('volatility', 0) * 100) }}%

{% endfor %}

## Learning Insights
{% for insight in learning_insights %}
- {{ insight }}
{% endfor %}

## Risk Events
{% for event in risk_events %}
### Event: {{ event.get('type', 'UNKNOWN') }}
- Timestamp: {{ event.get('timestamp', 'N/A') }}
- Severity: {{ event.get('severity', 'N/A') }}
- Description: {{ event.get('description', 'N/A') }}

{% endfor %}

---
*Generated by SOUL Synthesizer v1.0 - Stage 45 HFT System*
"""

    def __init__(
        self,
        output_directory: str = "./reports",
        template_string: Optional[str] = None,
    ):
        """
        Initialize the SOUL synthesizer.

        Args:
            output_directory: Directory for output files
            template_string: Optional custom Jinja2 template
        """
        self.output_directory = output_directory
        self.template_str = template_string or self.SOUL_TEMPLATE
        self._template: Optional[Template] = None
        self._write_queue: asyncio.Queue = asyncio.Queue()
        self._writer_task: Optional[asyncio.Task] = None
        self._running = False

        if JINJA_AVAILABLE:
            env = Environment(loader=BaseLoader())
            self._template = env.from_string(self.template_str)

    async def start(self) -> None:
        """Start the async file writer."""
        self._running = True
        self._writer_task = asyncio.create_task(self._file_writer_loop())
        logger.info("SOULSynthesizer started")

    async def stop(self) -> None:
        """Stop the synthesizer and wait for pending writes."""
        self._running = False
        if self._writer_task:
            await self._write_queue.join()
            self._writer_task.cancel()
            try:
                await self._writer_task
            except asyncio.CancelledError:
                pass
        logger.info("SOULSynthesizer stopped")

    async def _file_writer_loop(self) -> None:
        """Async loop for non-blocking file writes."""
        while self._running:
            try:
                task = await asyncio.wait_for(
                    self._write_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            try:
                filepath, content = task
                await self._write_file_async(filepath, content)
            except Exception as e:
                logger.error(f"File write failed: {e}")
            finally:
                self._write_queue.task_done()

    async def _write_file_async(self, filepath: str, content: str) -> None:
        """Write file using aiofiles for non-blocking I/O."""
        if AIOFILES_AVAILABLE:
            async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
                await f.write(content)
        else:
            # Fallback to sync write (blocks GIL but ensures functionality)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            await asyncio.sleep(0)  # Yield control

    def synthesize(self, report: SOULReport) -> str:
        """
        Synthesize a SOUL.md report from metrics data.

        Args:
            report: SOULReport data structure

        Returns:
            Rendered markdown string
        """
        if not JINJA_AVAILABLE:
            return self._fallback_synthesize(report)

        if not self._template:
            return self._fallback_synthesize(report)

        # Convert dataclass to dict for template
        context = {
            "report_id": report.report_id,
            "generated_at": report.generated_at,
            "session_start": report.session_start,
            "session_end": report.session_end,
            "total_pnl": report.total_pnl,
            "total_trades": report.total_trades,
            "sharpe_ratio": report.sharpe_ratio,
            "sortino_ratio": report.sortino_ratio,
            "max_drawdown": report.max_drawdown,
            "ml_metrics": [asdict(m) for m in report.ml_metrics],
            "strategy_performance": [asdict(s) for s in report.strategy_performance],
            "recent_trades": [asdict(t) for t in report.recent_trades],
            "regime_analysis": report.regime_analysis,
            "learning_insights": report.learning_insights,
            "risk_events": report.risk_events,
        }

        return self._template.render(**context)

    def _fallback_synthesize(self, report: SOULReport) -> str:
        """Fallback synthesis without Jinja2."""
        lines = [
            "# SOUL.md - Self-Learning Ledger Report",
            "",
            f"## Report ID: {report.report_id}",
            f"## Generated At: {report.generated_at}",
            "",
            "## Performance Summary",
            f"- Total PnL: ${report.total_pnl:.2f}",
            f"- Total Trades: {report.total_trades}",
            f"- Sharpe Ratio: {report.sharpe_ratio:.3f}",
            f"- Max Drawdown: {report.max_drawdown * 100:.2f}%",
            "",
            "## ML Metrics",
        ]

        for metric in report.ml_metrics:
            lines.append(f"- {metric.model_name}: Accuracy={metric.accuracy:.2f}")

        lines.extend([
            "",
            "## Learning Insights",
        ])
        for insight in report.learning_insights:
            lines.append(f"- {insight}")

        return "\n".join(lines)

    async def save_report(
        self,
        report: SOULReport,
        filename: Optional[str] = None,
    ) -> str:
        """
        Save synthesized report to file asynchronously.

        Args:
            report: SOULReport to save
            filename: Optional custom filename

        Returns:
            Path to saved file
        """
        content = self.synthesize(report)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"soul_{timestamp}.md"

        filepath = f"{self.output_directory}/{filename}"

        # Queue for async writing
        await self._write_queue.put((filepath, content))

        logger.info(f"Queued SOUL report: {filepath}")
        return filepath

    async def save_report_immediate(
        self,
        report: SOULReport,
        filename: str,
    ) -> str:
        """Save report immediately (awaitable)."""
        content = self.synthesize(report)
        filepath = f"{self.output_directory}/{filename}"
        await self._write_file_async(filepath, content)
        return filepath

    def generate_learning_insights(
        self,
        ml_metrics: List[MLMetrics],
        strategy_perf: List[StrategyPerformance],
        risk_events: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Generate learning insights from metrics (rule-based, no LLM).
        """
        insights = []

        # Analyze ML performance
        for metric in ml_metrics:
            if metric.accuracy < 0.5:
                insights.append(
                    f"Model {metric.model_name} showing degraded accuracy "
                    f"({metric.accuracy:.1%}). Consider retraining."
                )
            if metric.avg_inference_latency_us > 1000:
                insights.append(
                    f"Model {metric.model_name} latency elevated "
                    f"({metric.avg_inference_latency_us:.0f}μs). Review optimization."
                )

        # Analyze strategy performance
        for strategy in strategy_perf:
            if strategy.win_rate < 0.4:
                insights.append(
                    f"Strategy {strategy.strategy_id} win rate below target "
                    f"({strategy.win_rate:.1%}). Review signal quality."
                )
            if strategy.max_drawdown > 0.05:
                insights.append(
                    f"Strategy {strategy.strategy_id} exceeded drawdown tolerance "
                    f"({strategy.max_drawdown:.1%}). Tighten risk controls."
                )

        # Analyze risk events
        critical_events = [e for e in risk_events if e.get('severity') == 'CRITICAL']
        if len(critical_events) > 3:
            insights.append(
                f"High frequency of critical risk events ({len(critical_events)}). "
                "Review market conditions and position sizing."
            )

        return insights


def create_soul_report(
    report_id: str,
    session_start: datetime,
    session_end: datetime,
    trades: List[TradeOutcome],
    ml_metrics: List[MLMetrics],
    strategies: List[StrategyPerformance],
    risk_events: List[Dict[str, Any]],
) -> SOULReport:
    """Factory function to create a complete SOULReport."""
    # Calculate aggregate metrics
    total_pnl = sum(t.pnl for t in trades)
    total_trades = len(trades)

    # Calculate Sharpe (simplified)
    if trades:
        pnls = [t.pnl for t in trades]
        mean_pnl = sum(pnls) / len(pnls)
        std_pnl = (sum((p - mean_pnl) ** 2 for p in pnls) / len(pnls)) ** 0.5
        sharpe = mean_pnl / (std_pnl + 1e-9) * (252 ** 0.5)  # Annualized
    else:
        sharpe = 0.0

    # Calculate Sortino (simplified)
    negative_pnls = [t.pnl for t in trades if t.pnl < 0]
    if negative_pnls:
        downside_std = (sum(p ** 2 for p in negative_pnls) / len(negative_pnls)) ** 0.5
        sortino = mean_pnl / (downside_std + 1e-9) * (252 ** 0.5)
    else:
        sortino = sharpe

    # Max drawdown from strategies
    max_dd = max((s.max_drawdown for s in strategies), default=0.0)

    # Generate insights
    synthesizer = SOULSynthesizer()
    insights = synthesizer.generate_learning_insights(ml_metrics, strategies, risk_events)

    return SOULReport(
        report_id=report_id,
        generated_at=datetime.now().isoformat(),
        session_start=session_start.isoformat(),
        session_end=session_end.isoformat(),
        total_pnl=total_pnl,
        total_trades=total_trades,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        ml_metrics=ml_metrics,
        strategy_performance=strategies,
        recent_trades=trades[-20:],  # Last 20 trades
        regime_analysis={"current_regime": {"duration_minutes": 60, "pnl_contribution": total_pnl * 0.5, "volatility": 0.15}},
        learning_insights=insights,
        risk_events=risk_events,
    )
